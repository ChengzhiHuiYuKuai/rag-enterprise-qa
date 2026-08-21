"""混合检索策略：向量检索 + BM25"""

from loguru import logger
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.config import settings
from app.vectorstore.chroma import vector_store


class HybridRetriever:
    """混合检索器

    结合向量检索（语义相似）和 BM25（关键词匹配），
    通过加权融合提升检索质量。
    """

    def __init__(self):
        self._bm25_index: BM25Okapi | None = None
        self._bm25_docs: list[Document] = []
        self._bm25_tokenized: list[list[str]] = []

    def _build_bm25_index(self, docs: list[Document]) -> None:
        """构建 BM25 索引"""
        # 简单的中文分词：按字符 + 空格切分
        self._bm25_docs = docs
        self._bm25_tokenized = [self._tokenize(doc.page_content) for doc in docs]
        self._bm25_index = BM25Okapi(self._bm25_tokenized)
        logger.info(f"BM25 索引构建完成，共 {len(docs)} 个文档")

    def _ensure_index(self) -> None:
        """确保 BM25 索引已构建（从 ChromaDB 自动加载）"""
        if self._bm25_index is not None:
            return

        try:
            # 从 ChromaDB 加载所有文档
            collection = vector_store.client.get_collection(
                settings.chroma_collection_name
            )
            result = collection.get(include=["documents", "metadatas"])
            if result["documents"]:
                docs = [
                    Document(page_content=content, metadata=meta or {})
                    for content, meta in zip(result["documents"], result["metadatas"])
                ]
                self._build_bm25_index(docs)
                logger.info(f"BM25 索引从 ChromaDB 自动加载，共 {len(docs)} 个文档")
            else:
                logger.info("ChromaDB 中暂无文档，跳过 BM25 索引构建")
        except Exception as e:
            logger.warning(f"BM25 索引自动加载失败: {e}")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单的中英文分词

        对于生产环境建议使用 jieba 分词，
        这里用简单的字符级 + 空格切分保持轻量。
        """
        import re

        # 提取中文字符和英文单词
        chinese_chars = list(re.findall(r"[一-鿿]", text))
        english_words = re.findall(r"[a-zA-Z]+", text.lower())
        return chinese_chars + english_words

    def index_documents(self, docs: list[Document]) -> None:
        """为文档构建 BM25 索引（用于关键词检索）"""
        self._build_bm25_index(docs)

    def refresh_index(self) -> None:
        """刷新 BM25 索引（文档上传后调用）"""
        self._bm25_index = None
        self._ensure_index()

    def search(
        self,
        query: str,
        k: int | None = None,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> list[tuple[Document, float]]:
        """混合检索

        Args:
            query: 查询文本
            k: 返回结果数量
            bm25_weight: BM25 权重
            vector_weight: 向量检索权重

        Returns:
            (文档, 分数) 列表，分数越高越相关
        """
        k = k or settings.top_k

        # 确保 BM25 索引已构建
        self._ensure_index()
        bm25_w = bm25_weight or settings.bm25_weight
        vector_w = vector_weight or settings.vector_weight

        # 向量检索
        vector_results = vector_store.similarity_search_with_score(query, k=k * 2)
        # ChromaDB 返回的距离是 L2 距离，越小越相关，转换为相似度分数
        vector_scored = []
        for doc, distance in vector_results:
            similarity = 1.0 / (1.0 + distance)  # 转换为 0-1 的相似度
            vector_scored.append((doc, similarity))

        # BM25 检索
        bm25_scored = []
        if self._bm25_index is not None and self._bm25_docs:
            tokenized_query = self._tokenize(query)
            scores = self._bm25_index.get_scores(tokenized_query)
            # 取 top-k
            top_indices = scores.argsort()[-k * 2 :][::-1]
            for idx in top_indices:
                if scores[idx] > 0:
                    bm25_scored.append((self._bm25_docs[idx], float(scores[idx])))

        # 融合分数
        return self._merge_scores(vector_scored, bm25_scored, bm25_w, vector_w, k)

    @staticmethod
    def _merge_scores(
        vector_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
        bm25_weight: float,
        vector_weight: float,
        k: int,
    ) -> list[tuple[Document, float]]:
        """融合两路检索结果

        使用 Reciprocal Rank Fusion (RRF) 的变体：
        对每路结果归一化后加权求和。
        """
        # 归一化分数到 0-1
        def normalize(results: list[tuple[Document, float]]) -> dict:
            if not results:
                return {}
            max_score = max(s for _, s in results) or 1.0
            min_score = min(s for _, s in results)
            score_range = max_score - min_score or 1.0
            return {
                doc.page_content: ((s - min_score) / score_range, doc)
                for doc, s in results
            }

        vector_norm = normalize(vector_results)
        bm25_norm = normalize(bm25_results)

        # 合并
        all_contents = set(vector_norm.keys()) | set(bm25_norm.keys())
        merged = []
        for content in all_contents:
            v_score, v_doc = vector_norm.get(content, (0.0, None))
            b_score, b_doc = bm25_norm.get(content, (0.0, None))
            doc = v_doc or b_doc
            final_score = vector_weight * v_score + bm25_weight * b_score
            merged.append((doc, final_score))

        # 按分数降序排列
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]


# 全局单例
hybrid_retriever = HybridRetriever()
