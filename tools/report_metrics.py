"""
年度报告指标统计与 HTML 注入。

模块职责：
1) 从本地或 OSS 读取 CSV，统计指定指标；
2) 产出 {values, files} 结构，便于前端展示数值与 CSV 文件名；
3) 将统计结果注入 HTML（通过 <script id="report-metrics-data">）。

设计说明：
- 不写死具体卫星/时间：调用方可以通过参数或环境变量传入；
- 支持本地路径或 OSS key：OSS 会下载到本地缓存再读取；
- 只读取需要的列，减少大文件读取成本。
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config import PARQUET_ROOT as DATA_ROOT
from tools.oss_tool import _download_from_oss

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# 指标配置：
# - key: 前端 data-metric-key 对应的唯一键
# - bag_id: 包号（OSS/本地目录使用）
# - column: CSV 中的列名
# - mode: 统计方式（count_in / count_not_in / count_equals）
# - normal_values / abnormal_values / target_value: 与统计方式配套使用
METRIC_DEFS = [
    {
        "key": "whole_status",
        "bag_id": "0x0821",
        "column": "ZTMS021-整星状态",
        "mode": "count_in",
        "abnormal_values": ["不健康"],
    },
    {
        "key": "whole_collect_health",
        "bag_id": "0x0826",
        "column": "ZTMD211-整星采集健康状态",
        "mode": "count_in",
        "abnormal_values": ["不健康"],
    },
    {
        "key": "attitude_energy_safe",
        "bag_id": "0x0826",
        "column": "ZTMD208-转姿控能源安全标志",
        "mode": "count_not_in",
        "normal_values": ["能源充足"],
    },
    {
        "key": "energy_safe",
        "bag_id": "0x3030",
        "column": "能源安全",
        "mode": "count_in",
        "abnormal_values": ["发生过"],
    },
    {
        "key": "switch_count_motion",
        "bag_id": "0x0823",
        "column": "ZTMS229-星务自主请求切机计数(机动故障)",
        "mode": "count_equals",
        "target_value": 1,
    },
    {
        "key": "switch_count_collect",
        "bag_id": "0x0823",
        "column": "ZTMS229-星务自主请求切机计数(采集故障)",
        "mode": "count_equals",
        "target_value": 1,
    },
]

_ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "gbk", "ISO-8859-1"]
_WEEK_RE = re.compile(r"^(?P<year>\d{4})_(?P<week>\d{2})\.csv$")


def _read_csv(path: str, columns: Iterable[str]) -> Optional[pd.DataFrame]:
    """读取 CSV，按需挑选列并兼容多种编码。"""
    col_set = set(columns)
    usecols = lambda c: c in col_set
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, low_memory=False)
        except Exception:
            continue
    return None


def _normalize_week_file(week_file: Optional[str]) -> Optional[str]:
    """规范化文件名，允许传入 2025_01 或 2025_01.csv。"""
    if not week_file:
        return None
    week_file = week_file.strip()
    if not week_file:
        return None
    if not week_file.endswith(".csv"):
        return f"{week_file}.csv"
    return week_file


def _detect_satellite_from_local() -> Optional[str]:
    """从本地 data 目录里自动挑一个已有卫星目录。"""
    if not os.path.isdir(DATA_ROOT):
        return None
    candidates = [
        name for name in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, name))
    ]
    return sorted(candidates)[0] if candidates else None


def _pick_latest_week_file(local_dir: str) -> Optional[str]:
    """在本地目录里选取最新的 YYYY_WW.csv（按文件名排序）。"""
    if not os.path.isdir(local_dir):
        return None
    matches = []
    for name in os.listdir(local_dir):
        m = _WEEK_RE.match(name)
        if not m:
            continue
        matches.append((int(m.group("year")), int(m.group("week")), name))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][2]


def _resolve_source(source: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析数据源：
    - 绝对路径：直接使用
    - 相对路径：相对项目根目录
    - OSS key：下载到 data/<oss_key> 作为本地缓存
    """
    source = (source or "").strip()
    if not source:
        return None, "empty source"

    if os.path.isabs(source) and os.path.exists(source):
        return source, None

    local_path = os.path.join(PROJECT_ROOT, source)
    if os.path.exists(local_path):
        return local_path, None

    if not source.endswith(".csv"):
        return None, f"unsupported source: {source}"

    oss_key = source.lstrip("/")
    local_cache = os.path.join(DATA_ROOT, oss_key)
    if os.path.exists(local_cache):
        return local_cache, None

    success, msg = _download_from_oss(oss_key, local_cache)
    if not success:
        return None, msg
    return local_cache, None


def _count_in(series: pd.Series, target_values: List[str]) -> int:
    """统计列值属于 target_values 的数量（文本匹配）。"""
    s = series.dropna().astype(str).str.strip()
    target_set = set(str(v).strip() for v in target_values)
    return int(s.isin(target_set).sum())


def _count_not_in(series: pd.Series, normal_values: List[str]) -> int:
    """统计列值不属于 normal_values 的数量（文本匹配）。"""
    s = series.dropna().astype(str).str.strip()
    normal_set = set(str(v).strip() for v in normal_values)
    return int((~s.isin(normal_set)).sum())


def _count_equals(series: pd.Series, target_value) -> int:
    """统计列值等于 target_value 的数量（数值优先，失败再按文本）。"""
    if isinstance(target_value, (int, float)):
        s = pd.to_numeric(series, errors="coerce")
        return int((s == target_value).sum())
    s = series.dropna().astype(str).str.strip()
    return int((s == str(target_value)).sum())


def _metric_value(series: pd.Series, spec: Dict) -> int:
    """按配置选择统计方式。"""
    mode = spec.get("mode")
    if mode == "count_in":
        return _count_in(series, spec.get("abnormal_values", []))
    if mode == "count_not_in":
        return _count_not_in(series, spec.get("normal_values", []))
    if mode == "count_equals":
        return _count_equals(series, spec.get("target_value"))
    return 0


def _merge_file_name(existing: Optional[str], new_name: str) -> str:
    """合并文件名，避免重复（用于多文件统计场景）。"""
    if not existing:
        return new_name
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if new_name in parts:
        return existing
    parts.append(new_name)
    return ", ".join(parts)


def _build_sources_by_bag(
    satellite: Optional[str],
    week_file: Optional[str],
    bag_ids: List[str],
) -> Dict[str, List[str]]:
    """
    生成数据源映射（bag_id -> [source]）。
    - week_file 指定时：优先本地路径，否则使用 OSS key
    - week_file 未指定：尝试本地目录中最新文件
    """
    sources: Dict[str, List[str]] = {}
    satellite = satellite or _detect_satellite_from_local()
    if not satellite:
        return sources

    week_file = _normalize_week_file(week_file)
    for bag in bag_ids:
        if week_file:
            local_path = os.path.join(DATA_ROOT, satellite, bag, week_file)
            if os.path.exists(local_path):
                sources[bag] = [local_path]
            else:
                sources[bag] = [f"{satellite}/{bag}/{week_file}"]
            continue

        local_dir = os.path.join(DATA_ROOT, satellite, bag)
        latest = _pick_latest_week_file(local_dir)
        if latest:
            sources[bag] = [os.path.join(local_dir, latest)]
    return sources


def build_report_metrics(
    satellite: Optional[str] = None,
    week_file: Optional[str] = None,
    sources_by_bag: Optional[Dict[str, List[str]]] = None,
    metric_defs: Optional[List[Dict]] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    核心统计入口。

    参数说明：
    - satellite / week_file：用于自动构造数据源（不写死）
    - sources_by_bag：手动指定数据源（优先级最高）
    - metric_defs：自定义指标配置（默认使用 METRIC_DEFS）
    """
    metric_defs = metric_defs or METRIC_DEFS
    bag_ids = sorted({m["bag_id"] for m in metric_defs})
    sources_by_bag = sources_by_bag or _build_sources_by_bag(satellite, week_file, bag_ids)

    values: Dict[str, Optional[int]] = {m["key"]: None for m in metric_defs}
    files: Dict[str, Optional[str]] = {m["key"]: None for m in metric_defs}
    seen: Dict[str, bool] = {m["key"]: False for m in metric_defs}

    bag_to_metrics: Dict[str, List[Dict]] = {}
    for spec in metric_defs:
        bag_to_metrics.setdefault(spec["bag_id"], []).append(spec)

    for bag_id, bag_metrics in bag_to_metrics.items():
        sources = sources_by_bag.get(bag_id, [])
        if not sources:
            continue

        columns = [m["column"] for m in bag_metrics]
        for source in sources:
            path, err = _resolve_source(source)
            if not path:
                print(f"[report-metrics] skip {source}: {err}")
                continue

            df = _read_csv(path, columns)
            if df is None:
                print(f"[report-metrics] failed to read: {path}")
                continue

            file_name = os.path.basename(path)
            for spec in bag_metrics:
                key = spec["key"]
                files[key] = _merge_file_name(files.get(key), file_name)
                col = spec["column"]
                if col not in df.columns:
                    continue
                value = _metric_value(df[col], spec)
                prev = values[key] or 0
                values[key] = prev + value
                seen[key] = True

    for key, found in seen.items():
        if not found:
            values[key] = None

    return {
        "values": {k: ("-" if v is None else v) for k, v in values.items()},
        "files": {k: (files[k] or "-") for k in values.keys()},
        "meta": {
            "satellite": satellite or "",
            "week_file": _normalize_week_file(week_file) or "",
        },
    }


def inject_metrics_into_html(html_content: str, payload: Dict) -> str:
    """
    将统计结果注入 HTML。

    说明：
    - payload 必须包含 values/files 字段；
    - 若 HTML 中已存在 <script id="report-metrics-data">，则替换其内容；
    - 若不存在，则追加脚本（保证旧 HTML 也能用）。
    """
    # 兼容传入 values 字典的老调用方式
    if "values" not in payload:
        payload = {"values": payload, "files": {}}

    json_payload = json.dumps(payload, ensure_ascii=True)
    marker = '<script id="report-metrics-data" type="application/json">'

    if marker in html_content:
        pattern = r'(<script id="report-metrics-data" type="application/json">)(.*?)(</script>)'
        return re.sub(
            pattern,
            lambda m: m.group(1) + json_payload + m.group(3),
            html_content,
            flags=re.S,
        )

    script = (
        f"{marker}{json_payload}</script>"
        "<script>(function(){var el=document.getElementById('report-metrics-data');"
        "if(!el){return;}var data={};try{data=JSON.parse(el.textContent||'{}');}catch(e){}"
        "var values=data.values||{};var files=data.files||{};"
        "document.querySelectorAll('[data-metric-key]').forEach(function(node){"
        "var key=node.getAttribute('data-metric-key');"
        "if(Object.prototype.hasOwnProperty.call(values,key)){node.textContent=values[key];}"
        "});"
        "document.querySelectorAll('[data-metric-file]').forEach(function(node){"
        "var key=node.getAttribute('data-metric-file');"
        "if(Object.prototype.hasOwnProperty.call(files,key)){node.textContent=files[key];}"
        "});})();</script>"
    )
    if "</body>" in html_content:
        return html_content.replace("</body>", script + "</body>")
    return html_content + script
