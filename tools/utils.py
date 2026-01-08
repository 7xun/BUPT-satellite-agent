# -*- coding: utf-8 -*-
"""
通用辅助函数和类定义
"""
import ast
import operator as op
import os
from typing import List, Optional, Dict

import duckdb
import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL, 
    EMBEDDING_DIMENSIONS, VECTOR_STORE_DIR, BAG_NAME, BAG_LABEL, QWEN_MODEL
)

_RETRIEVER = None

def _require_key():
    if not DASHSCOPE_API_KEY or "REPLACE_ME" in DASHSCOPE_API_KEY:
        raise RuntimeError("请配置 DASHSCOPE_API_KEY")

def _normalize_bag_id(bag: str) -> str:
    """包名转 ID"""
    if bag in BAG_NAME:
        return BAG_LABEL[BAG_NAME.index(bag)]
    return bag

def _query_parquet(start_ns: int, end_ns: int, path: str, cols: List[str], index_col="time") -> pd.DataFrame:
    """DuckDB 查询 Parquet"""
    con = duckdb.connect()
    try:
        try:
            schema = con.sql(f"DESCRIBE SELECT * FROM '{path}'").df()
        except: return pd.DataFrame()

        all_cols = schema["column_name"].tolist()
        final_cols = {index_col} if index_col in all_cols else set()
        
        for target in cols:
            if target in all_cols: final_cols.add(target)
            else: # 前缀匹配
                for ac in all_cols:
                    if ac.startswith(target): final_cols.add(ac); break
        
        if not final_cols: return pd.DataFrame()

        select_str = ", ".join([f'"{c}"' for c in final_cols])
        df = con.sql(f"""SELECT {select_str} FROM '{path}' WHERE "{index_col}" >= {start_ns} AND "{index_col}" <= {end_ns}""").df()
        if "time" in df.columns: df["time"] = pd.to_datetime(df["time"], unit="ns")
        return df
    finally:
        con.close()

class DashScopeCompatibleEmbeddings(Embeddings):
    """兼容 DashScope 的 Embeddings"""
    def __init__(self, api_key, base_url, model, chunk_size=10, dimensions=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.chunk_size = chunk_size
        self.dimensions = dimensions

    def _embed_batch(self, batch):
        kwargs = {"model": self.model, "input": batch, "encoding_format": "float"}
        if self.dimensions: kwargs["dimensions"] = self.dimensions
        return [i.embedding for i in self.client.embeddings.create(**kwargs).data]

    def embed_documents(self, texts):
        clean = [str(t).strip() for t in texts if t and str(t).strip()]
        vectors = []
        for i in range(0, len(clean), self.chunk_size):
            vectors.extend(self._embed_batch(clean[i:i+self.chunk_size]))
        return vectors

    def embed_query(self, text):
        return self._embed_batch([str(text).strip()])[0] if str(text).strip() else []

def build_embeddings():
    _require_key()
    return DashScopeCompatibleEmbeddings(DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS)

def build_llm(model_name=QWEN_MODEL):
    _require_key()
    return ChatOpenAI(model=model_name, temperature=0, api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

def _get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        if not os.path.isdir(VECTOR_STORE_DIR):
            raise FileNotFoundError(f"未找到向量库: {VECTOR_STORE_DIR}\n请运行 build_index.py")
        vs = FAISS.load_local(VECTOR_STORE_DIR, build_embeddings(), allow_dangerous_deserialization=True)
        _RETRIEVER = vs.as_retriever(search_kwargs={"k": 4})
    return _RETRIEVER
