# -*- coding: utf-8 -*-
"""
LangServe 服务器启动脚本
"""
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from agent import build_runnable_with_history

app = FastAPI(
    title="卫星智能体 API",
    description="基于 LangChain 的卫星故障诊断智能体",
    version="1.0.0",
)

# 添加路由
add_routes(
    app,
    build_runnable_with_history(verbose=True),
    path="/chat",
    playground_type="chat",
)

if __name__ == "__main__":
    print("启动 LangServe: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
