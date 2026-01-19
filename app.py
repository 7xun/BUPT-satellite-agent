"""
Streamlit 前端应用。
"""
import functools
import http.server
import json
import os
import re
import threading
import time
from urllib.parse import quote as url_quote
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent import build_agent
from config import ANNUAL_REPORT_PATH, PARQUET_ROOT
from tools.oss_tool import _download_from_oss
from tools.report_metrics import METRIC_DEFS, build_report_metrics, inject_metrics_into_html
from tools.utils import build_llm


# --- 配置 ---

PAGE_TITLE = "北邮卫星智能体"
PAGE_ICON = "🤖"
LAYOUT = "wide"

CUSTOM_CSS = """
<style>
    .stChatFloatingInputContainer {bottom: 20px;}
    .block-container {padding-top: 2rem;}
    h1 {color: #0056b3; font-family: 'Segoe UI', sans-serif; font-weight: 600;}
    
    /* Sidebar Style */
    [data-testid="stSidebar"] {background-color: #f8f9fa; border-right: 1px solid #e9ecef;}
    
    /* Status Card */
    .status-card {
        background-color: #ffffff; 
        border-left: 4px solid #0056b3; 
        padding: 15px; 
        border-radius: 6px; 
        margin-bottom: 20px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .status-item {display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9em; color: #495057;}
    .status-item:last-child {margin-bottom: 0;}
    .status-value {font-weight: 600; color: #0056b3;}
    
    /* Button Style */
    .stButton button {
        border-radius: 8px; 
        border: 1px solid #dee2e6; 
        transition: all 0.2s;
        font-weight: 500;
    }
    .stButton button:hover {
        border-color: #0056b3; 
        color: #0056b3; 
        background-color: #e7f1ff;
        transform: translateY(-1px);
    }
</style>
"""


# --- 本地报告服务 ---

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return


@st.cache_resource
def _get_report_server(root_dir):
    handler = functools.partial(_QuietHandler, directory=root_dir)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# --- 辅助函数 ---

def init_session():
    """初始化会话状态。"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="👋 **您好！我是北邮卫星运维智能助手。**\n\n我可以帮您进行卫星体检、异常检测或查询故障知识库。")
        ]
    if "report_mode" not in st.session_state:
        st.session_state.report_mode = False


def load_agent(model_name):
    """加载或更新智能体。"""
    if "agent" not in st.session_state or st.session_state.get("current_model") != model_name:
        with st.spinner(f"正在切换模型到 {model_name}..."):
            st.session_state.agent = build_agent(verbose=True, model_name=model_name)
            st.session_state.current_model = model_name


def render_sidebar():
    """渲染侧边栏。"""
    with st.sidebar:
        st.title("⚙️ 控制面板")
        
        st.markdown("### 🤖 模型配置")
        model = st.radio("基础模型:", ("qwen-plus", "qwen3-omni-flash"), index=0)
        
        st.markdown("### 🖥️ 系统状态")
        msg_count = len(st.session_state.messages) // 2
        
        st.markdown(f"""
        <div class="status-card">
            <div class="status-item"><span>状态</span><span class="status-value">🟢 在线</span></div>
            <div class="status-item"><span>模型</span><span class="status-value">{model}</span></div>
            <div class="status-item"><span>知识库</span><span class="status-value">📚 已加载</span></div>
            <div class="status-item"><span>轮次</span><span class="status-value">{msg_count}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ 工具箱")
        if st.session_state.messages:
            chat_log = "\n\n".join([f"[{m.type.upper()}] {m.content}" for m in st.session_state.messages])
            st.download_button(
                "💾 导出日志", 
                chat_log, 
                file_name=f"chat_log_{int(time.time())}.txt", 
                use_container_width=True
            )
        
        if st.button("🗑️ 清除历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("---")
        st.caption("© 2025 北邮卫星团队")
            
    return model


def handle_action(prompt):
    """处理快捷操作。"""
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.rerun()


def extract_file_path(text, ext_pattern):
    """
    从文本中提取有效的文件路径。
    支持绝对路径、相对路径和文件名（含中文路径）。
    """
    patterns = [
        rf"((?:[a-zA-Z]:)?[\\/][^\s\"'<>]+\.{ext_pattern})",
        rf"(~[\\/][^\s\"'<>]+\.{ext_pattern})",
        rf"((?:\.\.?[\\/])[^\s\"'<>]+\.{ext_pattern})",
        rf"([\w\-.]+\.{ext_pattern})",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, re.IGNORECASE))

    for match in matches:
        path = match.strip().rstrip(".,;:，。；：")
        if os.path.exists(path):
            return path
        abs_path = os.path.abspath(os.path.expanduser(path))
        if os.path.exists(abs_path):
            return abs_path
            
    return None


def get_report_url(html_path):
    """为报告路径生成本地可访问的 URL。"""
    root_dir = os.path.dirname(html_path)
    filename = os.path.basename(html_path)
    server = _get_report_server(root_dir)
    port = server.server_address[1]
    return f"http://127.0.0.1:{port}/{url_quote(filename)}"


def _load_report_metrics(satellite=None, week_file=None, sources_by_bag=None):
    satellite = satellite or os.environ.get("REPORT_SATELLITE")
    week_file = week_file or os.environ.get("REPORT_WEEK")
    return build_report_metrics(
        satellite=satellite,
        week_file=week_file,
        sources_by_bag=sources_by_bag,
    )


def prepare_report_html(html_path, satellite=None, week_file=None, sources_by_bag=None):
    """注入报告指标并确保本地文件可被新标签页访问。"""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        payload = _load_report_metrics(
            satellite=satellite,
            week_file=week_file,
            sources_by_bag=sources_by_bag,
        )
        updated = inject_metrics_into_html(html_content, payload)
        if updated != html_content:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(updated)
        return updated
    except Exception as e:
        print(f"[report] 指标注入失败: {e}")
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""


def _report_prompt_text():
    return (
        "请告诉我要查看哪颗卫星、哪一年、第几周的报告。\n"
        "示例：E星 2025 第1周（等同于 2025_01）。\n"
        "输入“取消”可退出。"
    )


def _is_report_intent(text):
    if not text:
        return False
    normalized = re.sub(r"\s+", "", text)
    lowered = normalized.lower()
    if "lstm" in lowered or "异常检测" in normalized or "深度学习" in normalized:
        return False
    keywords = (
        "年度报告",
        "健康体检",
        "体检报告",
        "健康报告",
        "状态报告",
        "卫星状态报告",
    )
    if any(k in normalized for k in keywords):
        return True
    if "报告" in normalized and any(k in normalized for k in ("年度", "体检", "健康", "状态")):
        return True
    return False


def _has_full_report_params(text):
    if not text:
        return False
    sat = None
    m = re.search(r"([EFGH])\s*星", text, re.IGNORECASE)
    if not m:
        m = re.search(r"卫星\s*([EFGH])", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b([EFGH])\b", text, re.IGNORECASE)
    if m:
        sat = m.group(1).upper()

    year = None
    m = re.search(r"(\d{4})", text)
    if m:
        year = int(m.group(1))

    week = None
    m = re.search(r"(\d{1,2})\s*周", text)
    if m:
        week = int(m.group(1))
    else:
        m = re.search(r"_(\d{2})", text)
        if m:
            week = int(m.group(1))

    return bool(sat and year and week)


def _parse_report_request_llm(text, model_name):
    """用 LLM 将用户输入解析为结构化参数。"""
    system = (
        "你是参数解析器，只输出 JSON，不要解释。\n"
        "目标：从用户文本中提取 satellite/year/week。\n"
        "约束：satellite 只能是 E/F/G/H（允许 E星/卫星E 等）。\n"
        "week 必须是 1-53 的整数。\n"
        "如果无法提取或不合法，ok=false 并给出 reason。\n"
        "输出格式："
        '{"ok": true/false, "satellite": "E", "year": 2025, "week": 1, "reason": ""}'
    )
    llm = build_llm(model_name)
    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=text),
    ])
    return resp.content if hasattr(resp, "content") else str(resp)


def _parse_report_request(text, model_name):
    """优先 LLM 解析，失败时用规则兜底。"""
    raw = _parse_report_request_llm(text, model_name)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                return data if isinstance(data, dict) else {}
            except Exception:
                pass

    # 规则兜底
    sat = None
    m = re.search(r"([EFGH])\\s*星", text, re.IGNORECASE)
    if not m:
        m = re.search(r"卫星\\s*([EFGH])", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\\b([EFGH])\\b", text, re.IGNORECASE)
    if m:
        sat = m.group(1).upper()

    year = None
    m = re.search(r"(\\d{4})", text)
    if m:
        year = int(m.group(1))

    week = None
    m = re.search(r"(\\d{1,2})\\s*周", text)
    if m:
        week = int(m.group(1))
    else:
        m = re.search(r"_(\\d{2})", text)
        if m:
            week = int(m.group(1))

    return {"ok": bool(sat and year and week), "satellite": sat, "year": year, "week": week, "reason": "解析失败"}


def _validate_report_params(satellite, year, week):
    """校验解析结果合法性。"""
    if not satellite:
        return False, "未识别到卫星代号（仅支持 E/F/G/H）。"
    satellite = str(satellite).strip().upper()
    if satellite not in {"E", "F", "G", "H"}:
        return False, "卫星代号不合法，仅支持 E/F/G/H。"

    try:
        year = int(year)
    except Exception:
        return False, "年份格式不正确，应为 4 位数字。"
    if year < 1000 or year > 9999:
        return False, "年份超出合理范围。"

    try:
        week = int(week)
    except Exception:
        return False, "周次格式不正确，应为 1-53 的数字。"
    if week < 1 or week > 53:
        return False, "周次不合法，应为 1-53。"

    return True, {"satellite": satellite, "year": year, "week": week}


def _ensure_report_sources(satellite, week_file):
    """确保报告所需的 CSV 已下载到本地，返回 sources_by_bag 和缺失信息。"""
    sources_by_bag = {}
    missing = []
    bag_ids = sorted({m["bag_id"] for m in METRIC_DEFS})

    for bag in bag_ids:
        local_dir = os.path.join(PARQUET_ROOT, satellite, bag)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, week_file)
        if os.path.exists(local_path):
            sources_by_bag[bag] = [local_path]
            continue

        oss_key = f"{satellite}/{bag}/{week_file}"
        success, msg = _download_from_oss(oss_key, local_path)
        if success:
            sources_by_bag[bag] = [local_path]
        else:
            missing.append(f"{bag} ({msg})")

    return sources_by_bag, missing


def open_html_in_new_tab(html_path, opened_key):
    """最佳努力方式在新标签页打开 HTML。"""
    if "opened_report_tabs" not in st.session_state:
        st.session_state.opened_report_tabs = set()
    if opened_key in st.session_state.opened_report_tabs:
        return
    st.session_state.opened_report_tabs.add(opened_key)

    url = get_report_url(html_path)
    url_js = json.dumps(url)
    components.html(
        f"""
        <script>
        (function() {{
            const url = {url_js};
            const newWin = window.open(url, "_blank");
            if (newWin) {{
                if (newWin.blur) newWin.blur();
                if (window.focus) window.focus();
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def render_welcome():
    """渲染欢迎界面及操作按钮。"""
    st.markdown("### 💡 快速开始")
    st.markdown("选择一个任务或下方输入:")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📘 故障诊断\n\nGNSS 故障排查", use_container_width=True):
            handle_action("GNSS故障的一般步骤是什么？")
    with c2:
        if st.button("🔍 数据查询\n\nOSS 遥测数据", use_container_width=True):
            handle_action("帮我查询E卫星0x0821包中，2023年第7周的数据中‘ZTMS015-帆板1状态’字段值为‘未展开’的所有数据。")
    with c3:
        if st.button("📉 异常检测\n\n深度学习 (LSTM)", use_container_width=True):
            handle_action("运行lstm模型进行时序异常检测")
    with c4:
        if st.button("🏥 健康体检\n\n年度报告", use_container_width=True):
            st.session_state.report_mode = True
            st.session_state.messages.append(
                AIMessage(content=_report_prompt_text())
            )
            st.rerun()
    
    st.divider()


def render_chat():
    """渲染对话历史。"""
    for i, msg in enumerate(st.session_state.messages):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar = "🧑‍💻" if role == "user" else "🛰️"
        
        with st.chat_message(role, avatar=avatar):
            content = msg.content
            
            if role == "assistant":
                st.markdown(content)

                html_path = extract_file_path(content, "html")
                img_path = extract_file_path(content, "(?:png|jpg|jpeg)")
                json_path = extract_file_path(content, "json")

                if html_path and os.path.exists(html_path):
                    st.success(f"✅ 报告: {os.path.basename(html_path)}")
                    if os.path.abspath(html_path) == os.path.abspath(ANNUAL_REPORT_PATH):
                        params = st.session_state.get("report_params", {})
                        html_content = prepare_report_html(html_path, **params)
                    else:
                        with open(html_path, "r", encoding="utf-8") as f:
                            html_content = f.read()
                    open_html_in_new_tab(html_path, opened_key=f"report:{i}:{html_path}")
                    report_url = get_report_url(html_path)
                    st.markdown(
                        f'<a href="{report_url}" target="_blank">🔗 在新标签页打开报告</a>',
                        unsafe_allow_html=True,
                    )
                    components.iframe(report_url, height=900, scrolling=True)
                    st.download_button(
                        "📥 下载 HTML",
                        html_content.encode("utf-8"),
                        os.path.basename(html_path),
                        key=f"dl_html_{i}",
                    )

                elif img_path and os.path.exists(img_path):
                    st.success(f"✅ 图表: {os.path.basename(img_path)}")
                    st.image(img_path)
                    with open(img_path, "rb") as f:
                        st.download_button("📥 下载图片", f, os.path.basename(img_path), key=f"dl_img_{i}")

                elif json_path and os.path.exists(json_path):
                    st.success(f"✅ 数据: {os.path.basename(json_path)}")
                    try:
                        df = pd.read_json(json_path)
                        t1, t2 = st.tabs(["📈 图表", "📋 表格"])
                        with t1:
                            if "time" in df.columns:
                                plot_df = df.iloc[::len(df)//1000] if len(df) > 5000 else df
                                st.line_chart(plot_df.set_index("time").select_dtypes(include=['number']))
                            else:
                                st.info("未找到时间列。")
                        with t2:
                            st.dataframe(df)
                        with open(json_path, "rb") as f:
                            st.download_button("📥 下载 JSON", f, os.path.basename(json_path), key=f"dl_json_{i}")
                    except Exception as e:
                        st.error(f"读取数据失败: {e}")
            else:
                st.markdown(content)


def _handle_report_input(user_text, model_name):
    """处理年度报告流程中的用户输入。"""
    text = (user_text or "").strip()
    if not text:
        st.session_state.messages.append(AIMessage(content="请输入卫星与周次信息，例如：E星 2025 第1周。"))
        st.rerun()

    if text in {"取消", "退出", "算了", "不看了"}:
        st.session_state.report_mode = False
        st.session_state.messages.append(AIMessage(content="已退出年度报告查询。"))
        st.rerun()

    parsed = _parse_report_request(text, model_name)
    if not parsed:
        st.session_state.messages.append(
            AIMessage(content="未能解析到卫星/年份/周次，请按示例输入。\n\n示例：E星 2025 第1周")
        )
        st.rerun()

    ok_flag = parsed.get("ok") if isinstance(parsed, dict) else None
    if ok_flag is False:
        reason = parsed.get("reason") if isinstance(parsed, dict) else None
        reason = reason or "未能解析到卫星/年份/周次，请按示例输入。"
        st.session_state.messages.append(AIMessage(content=f"{reason}\n\n示例：E星 2025 第1周"))
        st.rerun()

    ok, data = _validate_report_params(
        parsed.get("satellite"),
        parsed.get("year"),
        parsed.get("week"),
    )
    if not ok:
        st.session_state.messages.append(AIMessage(content=f"{data}\n\n示例：E星 2025 第1周"))
        st.rerun()

    satellite = data["satellite"]
    year = data["year"]
    week = data["week"]
    week_file = f"{year}_{int(week):02d}.csv"

    sources_by_bag, missing = _ensure_report_sources(satellite, week_file)
    if not sources_by_bag:
        lines = [
            f"未找到任何可用数据（{satellite}星 {week_file}）。",
            "请确认卫星和周次是否正确。",
        ]
        if missing:
            lines.append("以下包下载失败：")
            lines.extend([f"- {item}" for item in missing])
        st.session_state.messages.append(
            AIMessage(
                content="\n".join(lines)
            )
        )
        st.rerun()

    html_path = ANNUAL_REPORT_PATH
    if not os.path.exists(html_path):
        st.session_state.report_mode = False
        st.session_state.messages.append(AIMessage(content=f"报告模板不存在: {html_path}"))
        st.rerun()

    prepare_report_html(
        html_path,
        satellite=satellite,
        week_file=week_file,
        sources_by_bag=sources_by_bag,
    )
    st.session_state.report_params = {
        "satellite": satellite,
        "week_file": week_file,
        "sources_by_bag": sources_by_bag,
    }

    lines = [
        f"已为您生成 {satellite} 星 {year} 年第 {int(week)} 周报告。",
        f"👉 报告路径：{html_path}",
    ]
    if missing:
        lines.append("以下包数据未获取到：")
        lines.extend([f"- {item}" for item in missing])

    st.session_state.report_mode = False
    st.session_state.messages.append(AIMessage(content="\n".join(lines)))
    st.rerun()


def process_input():
    """处理用户输入。"""
    # 处理待办操作
    if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
        user_text = st.session_state.messages[-1].content
        model_name = st.session_state.get("current_model", "qwen-plus")
        if st.session_state.get("report_mode"):
            with st.chat_message("assistant", avatar="🛰️"):
                with st.spinner("正在解析报告请求..."):
                    try:
                        _handle_report_input(user_text, model_name)
                    except Exception as e:
                        st.error(f"系统错误: {e}")
            return

        if _is_report_intent(user_text):
            st.session_state.report_mode = True
            if _has_full_report_params(user_text):
                with st.chat_message("assistant", avatar="🛰️"):
                    with st.spinner("正在解析报告请求..."):
                        try:
                            _handle_report_input(user_text, model_name)
                        except Exception as e:
                            st.error(f"系统错误: {e}")
                return

            st.session_state.messages.append(AIMessage(content=_report_prompt_text()))
            st.rerun()
            return

        with st.chat_message("assistant", avatar="🛰️"):
            with st.spinner("正在分析..."):
                try:
                    resp = st.session_state.agent.invoke({
                        "input": user_text,
                        "chat_history": st.session_state.messages[:-1]
                    })
                    st.session_state.messages.append(AIMessage(content=resp["output"]))
                    st.rerun()
                except Exception as e:
                    st.error(f"系统错误: {e}")
    
    # 仅在空闲时显示输入框
    if not (st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage)):
        if prompt := st.chat_input("输入指令..."):
            st.session_state.messages.append(HumanMessage(content=prompt))
            st.rerun()


# --- 主入口 ---

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session()
    model = render_sidebar()
    st.title("🛰️ 卫星运维智能体")
    
    load_agent(model)
    render_welcome()
    render_chat()
    process_input()


if __name__ == "__main__":
    main()
