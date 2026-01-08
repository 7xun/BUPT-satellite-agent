# 北邮卫星运维智能助手 (BUPT Satellite O&M Agent)

本项目是一个基于大语言模型（LLM）的卫星运维智能助手，旨在辅助卫星管理人员进行故障诊断、数据查询、异常检测和状态监控。项目集成了 LangChain、Streamlit、Aliyun OSS 以及深度学习异常检测模型。

## 📂 项目结构

### 核心模块
- **`app.py`**: Streamlit 前端应用入口。提供聊天界面、快捷指令按钮（故障查询、数据查询、异常检测、卫星体检）以及多媒体结果展示。
- **`agent.py`**: 智能体核心逻辑。定义了 LangChain Agent，集成了各类工具（Tools），并配置了 System Prompt。
- **`config.py`**: 全局配置文件。包含 API Key、OSS 配置、文件路径定义以及卫星包名映射。

### 工具集 (`tools/`)
- **`kb_tool.py`**: 知识库检索工具。基于 FAISS 向量数据库检索 `data_kb/` 中的故障文档。
- **`oss_tool.py`**: OSS 数据查询工具。通过自然语言生成 SQL 查询 Aliyun OSS 上的 CSV 数据。
- **`lstm_tool.py`**: 异常检测工具。调用 LSTM 模型对时序数据进行异常检测。
- **`report_tool.py`**: 报告获取工具。用于检索和展示卫星年度报告。
- **`utils.py`**: 通用辅助函数（如 LLM 构建、包名转换等）。

### 数据处理与存储 (`oss/` & `data/`)
- **`oss/`**: 包含与阿里云 OSS 交互的脚本。
    - `upload_to_oss.py`: 将本地数据上传至 OSS。
    - `query_oss_data.py`: OSS 数据查询逻辑实现。
    - `slice_csv_by_week.py`: 数据预处理脚本，将大文件按周切分。
    - `analyze_abnormal_data.py`: 异常数据分析脚本。
- **`data_kb/`**: 故障知识库文档（Markdown 格式）。
- **`index/`**: FAISS 向量索引文件，用于 RAG（检索增强生成）。
- **`frontend/`**: 存放生成的 HTML 报告及静态资源。

### 异常检测 (`prediction/`)
- 包含基于 TimeEval 框架的时间序列异常检测实验代码。
- **`run_lstm_ad_manual.py`**: 手动运行 LSTM 异常检测的脚本。
- **`lstm_ad_results.csv`**: 检测结果存储。
- **`plot_lstm_results.py`**: 结果可视化脚本。

---

## 🚀 快速开始

### 1. 环境准备

确保已安装 Python 3.8+ (建议使用 Conda 环境)。

```bash
# 安装依赖 (示例)
pip install streamlit langchain langchain-openai pandas oss2 duckdb torch faiss-cpu
# 或者根据具体报错安装缺失库
```

### 2. 配置项目

打开 `config.py`，确保以下配置正确：
- `DASHSCOPE_API_KEY`: 阿里云百炼/通义千问 API Key。
- `OSS_ACCESS_KEY_ID` / `SECRET`: 阿里云 OSS 访问凭证。
- `OSS_BUCKET_NAME`: 存储数据的 Bucket 名称。

### 3. 运行应用

在项目根目录下运行：

```bash
python -m streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`。

---

## 🧪 功能测试指南 (`prompt.txt`)

项目根目录下的 `prompt.txt` 文件包含了一些典型的测试指令，可用于验证各项功能是否正常。

### 测试步骤：

1.  **故障查询 (RAG)**
    *   **指令**: `我正在进行卫星故障排查。请检索知识库，告诉我 GNSS 故障诊断的一般步骤或常见现象有哪些？`
    *   **预期**: Agent 应检索 `data_kb` 中的文档，并总结 GNSS 故障的相关信息。

2.  **数据查询 (Text-to-SQL / OSS)**
    *   **指令**: `帮我查询E卫星0x0821包中，2023年第7周的数据中‘ZTMS015-帆板1状态’字段值为‘未展开’的所有数据。`
    *   **预期**: Agent 解析指令，调用 `oss_tool`，从 OSS 下载对应 CSV 片段，筛选数据并返回 JSON 或图表。

3.  **异常检测 (LSTM)**
    *   **指令**: `请运行 LSTM 模型对当前数据进行时序异常检测，并生成检测图表。`
    *   **预期**: Agent 调用 `run_lstm_ad`，运行模型，并展示生成的异常检测结果图。

4.  **结果分析**
    *   **指令**: `请分析一下刚才 LSTM 模型的检测结果，告诉我有哪些异常点。`
    *   **预期**: Agent 调用 `analyze_lstm_results`，读取 CSV 结果并用自然语言解释异常点。

5.  **卫星体检 (报告展示)**
    *   **指令**: `我想查看一下卫星的年度报告。`
    *   **预期**: Agent 返回 HTML 报告路径，前端自动渲染展示。

---

## 🛠️ 维护与扩展

- **添加新知识**: 将 Markdown 格式的故障文档放入 `data_kb/`，运行 `index/build_index.py` (需自行实现或确认构建逻辑) 重建索引。
- **添加新工具**: 在 `tools/` 下创建新脚本，使用 `@tool` 装饰器定义工具，并在 `agent.py` 的 `tools` 列表中注册。
