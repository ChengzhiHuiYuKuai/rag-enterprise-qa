"""对话相关数据模型"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(default="default", description="会话 ID")
    stream: bool = Field(default=True, description="是否流式响应")


class ChatResponse(BaseModel):
    """对话响应"""

    answer: str = Field(..., description="回答内容")
    session_id: str = Field(..., description="会话 ID")
    sources: list[dict] = Field(default_factory=list, description="引用的文档片段")


class SourceDocument(BaseModel):
    """检索到的源文档"""

    content: str = Field(..., description="文档片段内容")
    metadata: dict = Field(default_factory=dict, description="文档元数据")
    score: float = Field(default=0.0, description="相关性分数")
