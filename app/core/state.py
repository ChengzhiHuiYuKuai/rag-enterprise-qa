"""LangGraph 状态定义"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class RAGState(TypedDict):
    """RAG 问答系统的状态

    这是 LangGraph 状态图在节点之间传递的数据结构。
    每个节点读取状态、处理后更新状态。
    """

    # 对话消息历史（LangGraph 自动管理）
    messages: Annotated[list[BaseMessage], add_messages]

    # 用户原始问题
    question: str

    # 查询改写后的结果（用于多轮对话中的指代消解）
    rewritten_query: str

    # 检索到的文档片段
    retrieved_docs: list[dict]

    # 相关性判断结果：relevant / irrelevant
    relevance: str

    # 最终生成的回答
    answer: str

    # 引用的源文档
    sources: list[dict]

    # 会话 ID
    session_id: str
