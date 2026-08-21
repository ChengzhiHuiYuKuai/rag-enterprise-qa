"""LangGraph 状态图定义

精简流程，只保留核心 RAG 路径：

    用户提问 → 向量检索 → 生成回答
"""

from langgraph.graph import StateGraph, END

from app.core.state import RAGState
from app.core.nodes import (
    retrieve,
    generate_answer,
)


def build_rag_graph() -> StateGraph:
    """构建精简版 RAG 状态图

    流程：retrieve → generate
    只调用一次 LLM（生成回答），大幅提升响应速度。
    """
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_answer)

    # 定义边：检索 → 生成 → 结束
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# 全局单例
rag_graph = build_rag_graph()
