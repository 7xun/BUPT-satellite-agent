from langchain_core.tools import tool
from .utils import _get_retriever

@tool
def kb_search(query: str) -> str:
    """搜索本地知识库"""
    docs = _get_retriever().invoke(query)
    if not docs: return "未找到相关内容。"
    return "\n\n".join([f"[{i+1}] source={d.metadata.get('source','?')}\n{d.page_content}" for i, d in enumerate(docs)])
