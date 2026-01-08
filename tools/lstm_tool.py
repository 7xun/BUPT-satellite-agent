import os
import pandas as pd
from langchain_core.tools import tool
from .lstm_impl import run_lstm_detection_and_plot

@tool
def run_lstm_ad() -> str:
    """运行 LSTM 异常检测"""
    try: return f"检测完成，图表: {run_lstm_detection_and_plot()}"
    except Exception as e: return f"检测失败: {e}"

@tool
def analyze_lstm_results() -> str:
    """分析 LSTM 结果"""
    # 注意：这里假设 output 目录在当前工作目录下，或者使用 config 中的 OUTPUT_ROOT
    # 为了保持一致性，最好使用 config 中的路径，但原代码是硬编码的。
    # 让我们使用 config 中的 OUTPUT_ROOT，但需要注意原代码是 os.path.dirname(os.path.abspath(__file__))
    # 如果我们在 tools/lstm_tool.py 中使用 __file__，它会指向 tools 目录。
    # 原代码: os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "lstm_ad_results.csv")
    # 假设 agent.py 在根目录运行，output 也在根目录。
    # 我们可以使用 config.OUTPUT_ROOT
    from config import OUTPUT_ROOT
    
    csv_path = os.path.join(OUTPUT_ROOT, "lstm_ad_results.csv")
    if not os.path.exists(csv_path): return "未找到结果文件，请先运行检测。"
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'score' not in df.columns: return "数据无效。"
        
        stats = df['score'].describe()
        threshold = stats['mean'] + 3 * stats['std']
        anomalies = df[df['score'] > threshold]
        top5 = df.nlargest(5, 'score')[['timestamp', 'score']].to_string(index=False, header=False)
        
        return (f"**LSTM 分析报告**\n"
                f"- 时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}\n"
                f"- 异常分数: 均值 {stats['mean']:.4f}, Max {stats['max']:.4f}\n"
                f"- 异常点数: {len(anomalies)} (阈值 > {threshold:.4f})\n"
                f"- Top 5 异常:\n{top5}")
    except Exception as e: return f"分析失败: {e}"
