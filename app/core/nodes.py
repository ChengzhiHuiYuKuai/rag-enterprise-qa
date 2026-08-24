"""LangGraph 状态图节点实现"""

from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from app.core.state import RAGState
from app.core.retrieval import hybrid_retriever
from app.vectorstore.chroma import vector_store


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 字符/token）"""
    return max(1, len(text) // 2)


def _trim_history_by_tokens(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """按 token 预算从旧到新截断对话历史"""
    total = 0
    keep_from = len(messages)
    # 从最新消息往回累加，超出预算就截断最早的
    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = _estimate_tokens(messages[i].content)
        if total + msg_tokens > max_tokens:
            break
        total += msg_tokens
        keep_from = i
    trimmed = messages[keep_from:]
    if len(trimmed) < len(messages):
        logger.info(f"对话历史截断: {len(messages)} → {len(trimmed)} 条 (估算 {total} tokens)")
    return trimmed


_llm_instance = None


def get_llm():
    """获取 LLM 实例（单例模式）"""
    global _llm_instance
    if _llm_instance is None:
        from langchain_openai import ChatOpenAI
        from app.config import settings

        _llm_instance = ChatOpenAI(
            model=settings.mimo_model,
            base_url=settings.mimo_base_url,
            api_key=settings.mimo_api_key,
            temperature=0.3,
            streaming=True,
        )
    return _llm_instance


def rewrite_query(state: RAGState) -> dict:
    """查询改写节点

    在多轮对话中，用户的问题可能包含指代（如"它"、"这个"），
    此节点利用对话历史将问题改写为独立、完整的查询。
    """
    question = state["question"]
    messages = state.get("messages", [])

    # 如果是第一轮对话或历史很短，直接使用原始问题
    if len(messages) <= 1:
        logger.info(f"首轮对话，跳过查询改写: {question}")
        return {"rewritten_query": question}

    # 构建改写 prompt
    history_text = ""
    for msg in messages[-6:]:  # 最近 3 轮对话
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        history_text += f"{role}: {msg.content}\n"

    prompt = [
        SystemMessage(
            content="你是一个查询改写助手。根据对话历史，将用户的最新问题改写为一个独立、完整的查询。"
            "如果问题已经是独立的，直接返回原问题。只返回改写后的查询，不要解释。"
        ),
        HumanMessage(
            content=f"对话历史:\n{history_text}\n最新问题: {question}\n\n改写后的查询:"
        ),
    ]

    llm = get_llm()
    response = llm.invoke(prompt)
    rewritten = response.content.strip()

    logger.info(f"查询改写: '{question}' → '{rewritten}'")
    return {"rewritten_query": rewritten}


def retrieve(state: RAGState) -> dict:
    """检索节点

    使用混合检索策略（向量 + BM25）查找相关文档片段。
    """
    from app.config import settings

    query = state.get("rewritten_query") or state["question"]

    # 使用混合检索（向量 + BM25 加权融合）
    results = hybrid_retriever.search(query, k=settings.top_k)

    docs = []
    for doc, score in results:
        docs.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
        )

    logger.info(f"检索完成，返回 {len(docs)} 个文档片段")
    return {"retrieved_docs": docs}


def grade_relevance(state: RAGState) -> dict:
    """相关性判断节点

    用 LLM 判断检索到的文档是否与问题相关，
    决定是直接生成回答还是走兜底逻辑。
    """
    question = state.get("rewritten_query") or state["question"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"relevance": "irrelevant"}

    # 拼接检索结果
    docs_text = "\n\n---\n\n".join(
        [f"[片段 {i+1}] {d['content']}" for i, d in enumerate(docs[:3])]
    )

    prompt = [
        SystemMessage(
            content="你是一个相关性判断助手。判断检索到的文档片段是否能回答用户的问题。"
            "如果文档与问题相关，回答 'relevant'；如果不相关，回答 'irrelevant'。"
            "只回答一个词。"
        ),
        HumanMessage(
            content=f"用户问题: {question}\n\n检索到的文档:\n{docs_text}\n\n是否相关:"
        ),
    ]

    llm = get_llm()
    response = llm.invoke(prompt)
    relevance = response.content.strip().lower()

    if "irrelevant" in relevance:
        relevance = "irrelevant"
    else:
        relevance = "relevant"

    logger.info(f"相关性判断: {relevance}")
    return {"relevance": relevance}


def generate_answer(state: RAGState) -> dict:
    """生成回答节点

    根据检索到的文档和对话历史生成回答。
    """
    from app.config import settings

    question = state.get("rewritten_query") or state["question"]
    docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])

    # 构建上下文
    context = "\n\n---\n\n".join(
        [f"[来源: {d['metadata'].get('source', '未知')}] {d['content']}" for d in docs]
    )

    # 按 token 预算截断对话历史（排除最后一条当前问题）
    history_msgs = messages[:-1] if messages else []
    history = _trim_history_by_tokens(history_msgs, settings.max_history_tokens)

    prompt = [
        SystemMessage(
            content="你是一个企业知识库问答助手。根据提供的文档上下文回答用户的问题。"
            "要求：\n"
            "1. 只根据提供的上下文回答，不要编造信息\n"
            "2. 如果上下文中没有相关信息，如实告知\n"
            "3. 回答要准确、简洁、专业\n"
            "4. 在回答末尾标注引用的来源"
        ),
        *history,
        HumanMessage(
            content=f"文档上下文:\n{context}\n\n用户问题: {question}\n\n请回答:"
        ),
    ]

    llm = get_llm()
    response = llm.invoke(prompt)
    answer = response.content

    # 整理来源信息
    sources = []
    for doc in docs:
        sources.append(
            {
                "content": doc["content"][:200],
                "source": doc["metadata"].get("source", "未知"),
                "score": doc.get("score", 0),
            }
        )

    logger.info(f"生成回答完成，引用 {len(sources)} 个来源")
    return {"answer": answer, "sources": sources}


def fallback(state: RAGState) -> dict:
    """兜底节点

    当检索结果不相关时，给出友好的提示。
    """
    question = state["question"]

    prompt = [
        SystemMessage(
            content="你是一个企业知识库问答助手。用户的问题在知识库中没有找到相关信息。"
            "请友好地告知用户，并建议他们尝试换一种方式提问，或者联系管理员上传相关文档。"
        ),
        HumanMessage(content=f"用户问题: {question}"),
    ]

    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [],
    }


def route_after_grade(state: RAGState) -> str:
    """相关性判断后的路由

    根据检索结果的相关性，决定是生成回答还是走兜底逻辑。
    """
    relevance = state.get("relevance", "irrelevant")
    if relevance == "relevant":
        return "generate"
    else:
        return "fallback"
