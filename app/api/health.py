"""健康检查接口"""

from fastapi import APIRouter

from app.vectorstore.chroma import vector_store
from app.config import settings

router = APIRouter(tags=["系统"])


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model": settings.mimo_model,
        "embedding_model": settings.embedding_model,
    }


@router.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    stats = vector_store.get_collection_stats()
    return {
        "vector_store": stats,
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "bm25_weight": settings.bm25_weight,
            "vector_weight": settings.vector_weight,
        },
    }
