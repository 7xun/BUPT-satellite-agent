# -*- coding: utf-8 -*-
"""
智能体核心模块
"""
from __future__ import annotations

from typing import Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from config import QWEN_MODEL
from tools.utils import build_llm
from tools import (
    get_satellite_status_report, 
    kb_search, 
    run_lstm_ad, 
    analyze_lstm_results,
    query_oss_csv_data
)

_STORE: Dict[str, ChatMessageHistory] = {}

# =============================================================================
# 智能体构建
# =============================================================================

def build_agent(verbose=False, model_name=QWEN_MODEL):
    llm = build_llm(model_name)
    tools = [get_satellite_status_report, kb_search, run_lstm_ad, analyze_lstm_results, query_oss_csv_data]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "你是一个卫星故障诊断助手。规则：\n"
         "1. 问“状态/报告” -> get_satellite_status_report (直接返回路径)\n"
         "2. 问“知识/文档” -> kb_search (引用来源)\n"
         "3. 问“LSTM/异常检测” -> run_lstm_ad\n"
         "4. 问“分析结果/异常点” -> analyze_lstm_results\n"
         "5. 问“OSS数据/CSV查询” -> query_oss_csv_data\n"
         "   示例: 查询E卫星0x0821包2023年第7周ZTMS015-帆板1状态为未展开的数据\n"
         "6. 展示图片/JSON -> 直接在回答中包含文件路径\n"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    return AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=verbose)

def build_runnable_with_history(verbose=False, model_name=QWEN_MODEL):
    return RunnableWithMessageHistory(
        build_agent(verbose, model_name),
        lambda sid: _STORE.setdefault(sid, ChatMessageHistory()),
        input_messages_key="input",
        history_messages_key="chat_history",
    )

if __name__ == "__main__":
    agent = build_agent(verbose=True)
    print("智能体启动 (输入 exit 退出)")
    while True:
        try:
            q = input("\n你: ").strip()
            if q.lower() in {"exit", "quit"}: break
            if q: print("\n助手:", agent.invoke({"input": q, "chat_history": []})["output"])
        except KeyboardInterrupt: break
        except Exception as e: print("错误:", e)
