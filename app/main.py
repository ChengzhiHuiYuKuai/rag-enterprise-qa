"""FastAPI 应用入口"""

# 必须在所有其他导入之前加载环境变量
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import chat, documents, health
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 RAG 企业知识库问答系统启动中...")
    logger.info(f"   LLM: {settings.mimo_model} @ {settings.mimo_base_url}")
    logger.info(f"   Embedding: {settings.embedding_model} ({settings.embedding_device})")
    logger.info(f"   ChromaDB: {settings.chroma_persist_dir}")
    yield
    logger.info("👋 系统关闭")


app = FastAPI(
    title="RAG 企业知识库问答系统",
    description="基于 LangChain + LangGraph 的 RAG 问答系统，支持多格式文档上传和智能问答",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
