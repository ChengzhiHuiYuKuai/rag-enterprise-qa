"""对话 API 接口

支持普通响应和 SSE 流式响应（token 级真正的流式输出）。
"""

import json
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from app.models.chat import ChatRequest, ChatResponse
from app.core.graph import rag_graph
from app.config import settings

router = APIRouter(prefix="/chat", tags=["对话"])

# 会话内存存储（生产环境建议用 Redis）
_sessions: dict[str, list] = {}

# 存储历史的 token 预算（留余量，实际发给 LLM 的由 nodes.py 再截一次）
_STORE_TOKEN_BUDGET = settings.max_history_tokens * 2


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数"""
    return max(1, len(text) // 2)


def _trim_session_history(messages: list, max_tokens: int) -> list:
    """按 token 预算截断会话存储的历史（从旧到新保留）"""
    total = 0
    keep_from = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = _estimate_tokens(messages[i].content)
        if total + msg_tokens > max_tokens:
            break
        total += msg_tokens
        keep_from = i
    return messages[keep_from:]


def _sse_event(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口

    支持普通和流式两种模式：
    - stream=false: 返回完整 JSON 响应
    - stream=true: 返回 SSE 流式响应（token 级流式输出）
    """
    session_id = request.session_id or str(uuid4())

    # 获取会话历史
    history = _sessions.get(session_id, [])

    # 构建初始状态
    initial_state = {
        "question": request.question,
        "messages": history + [HumanMessage(content=request.question)],
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevance": "",
        "answer": "",
        "sources": [],
        "session_id": session_id,
    }

    if request.stream:
        return StreamingResponse(
            _stream_response(initial_state, session_id),
            media_type="text/event-stream",
        )

    # 非流式：直接执行图
    result = await rag_graph.ainvoke(initial_state)

    # 更新会话历史（按 token 预算截断）
    history.append(HumanMessage(content=request.question))
    history.append(AIMessage(content=result["answer"]))
    _sessions[session_id] = _trim_session_history(history, _STORE_TOKEN_BUDGET)

    return ChatResponse(
        answer=result["answer"],
        session_id=session_id,
        sources=result.get("sources", []),
    )


async def _stream_response(initial_state: dict, session_id: str):
    """SSE 流式响应生成器（token 级真正的流式输出）

    使用 LangGraph 的 astream_events 捕获 LLM 逐 token 的流式输出，
    而非等待节点完成后一次性返回。

    事件类型：
    - metadata: 检索完成，返回文档数量
    - token: LLM 生成的每个 token（真正的流式）
    - sources: 引用的源文档
    - done: 完成信号
    - error: 异常信息
    """
    full_answer = ""
    sources = []

    try:
        # 使用 astream_events 获取细粒度事件（包括 LLM token 流）
        async for event in rag_graph.astream_events(initial_state, version="v2"):
            kind = event["event"]

            # 检索节点完成 → 发送检索元数据
            if kind == "on_chain_end" and event.get("name") == "retrieve":
                output = event.get("data", {}).get("output", {})
                docs = output.get("retrieved_docs", [])
                yield _sse_event("metadata", {"retrieved_count": len(docs)})

            # LLM token 流式输出（真正的逐 token 流）
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", {})
                content = getattr(chunk, "content", "")
                if content:
                    full_answer += content
                    yield _sse_event("token", {"token": content})

            # generate 节点完成 → 发送 sources
            elif kind == "on_chain_end" and event.get("name") == "generate":
                output = event.get("data", {}).get("output", {})
                sources = output.get("sources", [])
                yield _sse_event("sources", {"sources": sources})

        # 更新会话历史（按 token 预算截断）
        history = _sessions.get(session_id, [])
        history.append(HumanMessage(content=initial_state["question"]))
        history.append(AIMessage(content=full_answer))
        _sessions[session_id] = _trim_session_history(history, _STORE_TOKEN_BUDGET)

        yield _sse_event("done", {"session_id": session_id})

    except Exception as e:
        logger.error(f"流式响应异常: {e}")
        yield _sse_event("error", {"error": str(e)})


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清除会话历史"""
    _sessions.pop(session_id, None)
    return {"message": f"会话 {session_id} 已清除"}


@router.get("/sessions")
async def list_sessions():
    """列出所有活跃会话"""
    return {
        "sessions": [
            {"session_id": sid, "turns": len(msgs) // 2}
            for sid, msgs in _sessions.items()
        ]
    }
