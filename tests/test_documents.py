"""文档处理模块测试"""

from app.document.loader import DocumentLoader
from app.document.splitter import split_documents


def test_supported_extensions():
    """测试文件格式支持检查"""
    assert DocumentLoader.is_supported("test.pdf")
    assert DocumentLoader.is_supported("test.docx")
    assert DocumentLoader.is_supported("test.md")
    assert DocumentLoader.is_supported("test.xlsx")
    assert not DocumentLoader.is_supported("test.txt")
    assert not DocumentLoader.is_supported("test.csv")


def test_splitter_basic():
    """测试文档切分"""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="这是一段很长的文本。" * 100,
            metadata={"source": "test.md"},
        )
    ]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # 允许略微超出
        assert "source" in chunk.metadata
