"""文档切分策略"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from app.config import settings


def get_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    """获取文本切分器"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # 段落
            "\n",    # 换行
            "。",    # 中文句号
            "！",    # 中文感叹号
            "？",    # 中文问号
            ". ",    # 英文句号
            " ",     # 空格
            "",      # 字符级
        ],
    )


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """将文档切分为更小的块

    Args:
        docs: 原始文档列表
        chunk_size: 块大小，默认从配置读取
        chunk_overlap: 重叠大小，默认从配置读取

    Returns:
        切分后的文档列表，每个块保留原始 metadata 并添加 chunk 序号
    """
    splitter = get_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)

    # 为每个 chunk 添加序号
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        f"文档切分完成: {len(docs)} 个文档 → {len(chunks)} 个块 "
        f"(chunk_size={chunk_size or settings.chunk_size}, "
        f"overlap={chunk_overlap or settings.chunk_overlap})"
    )
    return chunks
