"""ChromaDB 向量存储管理"""

from pathlib import Path
from loguru import logger

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.vectorstore.embeddings import get_embeddings


class VectorStore:
    """ChromaDB 向量存储封装

    使用嵌入式模式，数据持久化到本地目录。
    支持文档的增删查，以及向量检索。
    """

    def __init__(self):
        self._client: chromadb.ClientAPI | None = None
        self._store: Chroma | None = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            persist_dir = settings.chroma_persist_dir
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"ChromaDB 初始化完成，持久化目录: {persist_dir}")
        return self._client

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                client=self.client,
                collection_name=settings.chroma_collection_name,
                embedding_function=get_embeddings(),
            )
            logger.info(f"ChromaDB 集合: {settings.chroma_collection_name}")
        return self._store

    def add_documents(self, documents: list[Document], doc_id: str) -> int:
        """添加文档到向量存储

        Args:
            documents: 切分后的文档列表
            doc_id: 文档唯一标识，用于后续管理和删除

        Returns:
            添加的文档块数量
        """
        # 在 metadata 中记录 doc_id，方便按文档删除
        for doc in documents:
            doc.metadata["doc_id"] = doc_id

        self.store.add_documents(documents)
        logger.info(f"文档 {doc_id} 已添加到向量存储，共 {len(documents)} 个块")
        return len(documents)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """向量相似度检索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相关文档列表
        """
        k = k or settings.top_k
        results = self.store.similarity_search(query, k=k)
        return results

    def similarity_search_with_score(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """带分数的向量相似度检索"""
        k = k or settings.top_k
        results = self.store.similarity_search_with_score(query, k=k)
        return results

    def delete_by_doc_id(self, doc_id: str) -> None:
        """按文档 ID 删除所有相关块"""
        collection = self.client.get_collection(settings.chroma_collection_name)
        collection.delete(where={"doc_id": doc_id})
        logger.info(f"已删除文档 {doc_id} 的所有向量")

    def get_doc_ids(self) -> list[str]:
        """获取所有已存储的文档 ID"""
        collection = self.client.get_collection(settings.chroma_collection_name)
        result = collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in result["metadatas"]:
            if "doc_id" in meta:
                doc_ids.add(meta["doc_id"])
        return list(doc_ids)

    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        collection = self.client.get_collection(settings.chroma_collection_name)
        return {
            "total_chunks": collection.count(),
            "collection_name": settings.chroma_collection_name,
        }


# 全局单例
vector_store = VectorStore()
