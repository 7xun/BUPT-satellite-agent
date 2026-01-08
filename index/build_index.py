# -*- coding: utf-8 -*-
"""
知识库索引构建脚本
"""
import os
import sys
from pathlib import Path
from typing import List

# 将项目根目录添加到 sys.path，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from config import VECTOR_STORE_DIR, DATA_KB_DIR, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL

# 配置
DATA_DIR = Path(DATA_KB_DIR)
INDEX_DIR = Path(VECTOR_STORE_DIR)

def load_documents() -> List[Document]:
    """加载 data_kb 目录下的 PDF/TXT/MD 文档"""
    docs = []
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"目录不存在: {DATA_DIR}")
        
    for p in DATA_DIR.rglob("*"):
        if p.is_dir(): continue
        try:
            if p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            elif p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
        except Exception as e:
            print(f"加载失败 {p.name}: {e}")
    return docs

def main():
    if not DASHSCOPE_API_KEY:
        raise ValueError("未配置 DASHSCOPE_API_KEY")

    print("正在加载文档...")
    raw_docs = load_documents()
    if not raw_docs:
        print("未找到文档，请检查 data_kb 目录")
        return

    print(f"加载 {len(raw_docs)} 个文档，正在切分...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.split_documents(raw_docs)
    docs = [d for d in docs if d.page_content.strip()]

    print(f"生成 {len(docs)} 个切片，正在构建索引...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )

    vs = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    print(f"✅ 索引构建完成: {INDEX_DIR.resolve()}")

if __name__ == "__main__":
    main()
