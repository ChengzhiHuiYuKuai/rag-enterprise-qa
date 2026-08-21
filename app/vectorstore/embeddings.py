"""Embedding 模型管理"""

import os
from loguru import logger
from langchain_core.embeddings import Embeddings


class BGEEmbeddings(Embeddings):
    """BGE-M3 Embedding 模型封装

    使用 sentence-transformers 加载本地模型，支持 GPU 推理。
    首次运行会自动从 HuggingFace 下载模型。
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            logger.info(f"加载 Embedding 模型: {self.model_name} (device={self.device})")
            from sentence_transformers import SentenceTransformer

            # 使用 HuggingFace 镜像（国内网络需要）
            hf_endpoint = os.getenv("HF_ENDPOINT", "")
            if hf_endpoint:
                os.environ["HF_ENDPOINT"] = hf_endpoint
                logger.info(f"使用 HuggingFace 镜像: {hf_endpoint}")

            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding 模型加载完成")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档"""
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本"""
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


def get_embeddings() -> BGEEmbeddings:
    """获取 Embedding 模型实例"""
    from app.config import settings

    return BGEEmbeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
