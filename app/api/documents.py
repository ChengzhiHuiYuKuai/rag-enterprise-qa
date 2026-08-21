"""文档管理 API 接口"""

import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.config import settings
from app.document.loader import DocumentLoader
from app.document.splitter import split_documents
from app.vectorstore.chroma import vector_store
from app.core.retrieval import hybrid_retriever
from app.models.document import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
)

router = APIRouter(prefix="/documents", tags=["文档管理"])

# 文档元数据持久化
_registry_path = Path(settings.upload_dir).parent / "doc_registry.json"


def _load_registry() -> dict[str, DocumentInfo]:
    """从 JSON 文件加载文档注册表"""
    if _registry_path.exists():
        try:
            with open(_registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: DocumentInfo(**v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"加载文档注册表失败: {e}")
    return {}


def _save_registry(registry: dict[str, DocumentInfo]) -> None:
    """保存文档注册表到 JSON 文件"""
    try:
        _registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_registry_path, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in registry.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存文档注册表失败: {e}")


# 启动时加载
_doc_registry: dict[str, DocumentInfo] = _load_registry()
logger.info(f"已加载 {_doc_registry.__len__()} 个文档元数据")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传并处理文档

    支持格式: PDF, Word (.docx), Markdown (.md), Excel (.xlsx)
    """
    # 校验文件格式
    if not DocumentLoader.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持: {', '.join(DocumentLoader.SUPPORTED_EXTENSIONS)}",
        )

    # 保存文件
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid4())[:8]
    file_ext = Path(file.filename).suffix
    save_path = upload_dir / f"{doc_id}{file_ext}"

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # 加载文档
        docs = DocumentLoader.load(str(save_path))

        # 切分文档
        chunks = split_documents(docs)

        # 存入向量数据库
        chunk_count = vector_store.add_documents(chunks, doc_id=doc_id)

        # 记录元数据
        doc_info = DocumentInfo(
            doc_id=doc_id,
            filename=file.filename,
            chunk_count=chunk_count,
            file_size=len(content),
        )
        _doc_registry[doc_id] = doc_info
        _save_registry(_doc_registry)

        # 刷新 BM25 索引
        hybrid_retriever.refresh_index()

        logger.info(f"文档上传成功: {file.filename} → {chunk_count} 个块")

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunk_count=chunk_count,
        )

    except Exception as e:
        # 清理失败的文件
        save_path.unlink(missing_ok=True)
        logger.error(f"文档处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """列出所有已上传的文档"""
    return DocumentListResponse(
        documents=list(_doc_registry.values()),
        total=len(_doc_registry),
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """获取文档详情"""
    if doc_id not in _doc_registry:
        raise HTTPException(status_code=404, detail="文档不存在")
    return _doc_registry[doc_id]


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    """删除文档

    同时删除向量数据库中的数据和本地文件。
    """
    if doc_id not in _doc_registry:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 从向量数据库删除
    try:
        vector_store.delete_by_doc_id(doc_id)
    except Exception as e:
        logger.warning(f"向量删除异常: {e}")

    # 删除本地文件
    upload_dir = Path(settings.upload_dir)
    for f in upload_dir.glob(f"{doc_id}.*"):
        f.unlink(missing_ok=True)

    # 删除元数据
    doc_info = _doc_registry.pop(doc_id)
    _save_registry(_doc_registry)

    # 刷新 BM25 索引
    hybrid_retriever.refresh_index()

    logger.info(f"文档已删除: {doc_info.filename}")

    return DocumentDeleteResponse(doc_id=doc_id)
