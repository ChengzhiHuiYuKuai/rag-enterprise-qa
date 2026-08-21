"""应用配置管理"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """从 .env 文件或环境变量加载配置"""

    # LLM
    mimo_api_key: str = Field(..., description="mimo API Key")
    mimo_base_url: str = Field(
        default="https://api.xiaomimimo.com/v1", description="mimo API endpoint"
    )
    mimo_model: str = Field(default="mimo-v2.5-pro", description="模型名称")

    # Embedding
    embedding_model: str = Field(
        default="BAAI/bge-m3", description="Embedding 模型名称"
    )
    embedding_device: str = Field(default="cuda", description="推理设备 cuda/cpu")

    # ChromaDB
    chroma_persist_dir: str = Field(
        default=str(BASE_DIR / "data" / "chroma"), description="ChromaDB 持久化目录"
    )
    chroma_collection_name: str = Field(
        default="enterprise_docs", description="集合名称"
    )

    # Document Processing
    chunk_size: int = Field(default=512, description="文档切分块大小(字符数)")
    chunk_overlap: int = Field(default=64, description="切分块重叠大小")

    # Retrieval
    top_k: int = Field(default=5, description="检索返回的文档块数量")
    bm25_weight: float = Field(default=0.3, description="BM25 检索权重")
    vector_weight: float = Field(default=0.7, description="向量检索权重")

    # Conversation Memory
    max_history_tokens: int = Field(default=2000, description="对话历史最大 token 数")
    summary_threshold: int = Field(
        default=10, description="超过此轮数触发摘要压缩"
    )

    # Server
    api_host: str = Field(default="0.0.0.0", description="API 服务地址")
    api_port: int = Field(default=8000, description="API 服务端口")
    streamlit_port: int = Field(default=8501, description="Streamlit 端口")

    # Upload
    upload_dir: str = Field(
        default=str(BASE_DIR / "data" / "uploads"), description="上传文件存储目录"
    )

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
