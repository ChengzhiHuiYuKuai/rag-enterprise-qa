"""文档相关数据模型"""

from datetime import datetime
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""

    doc_id: str = Field(..., description="文档 ID")
    filename: str = Field(..., description="文件名")
    chunk_count: int = Field(..., description="切分后的文档块数量")
    message: str = Field(default="文档上传并处理成功", description="提示信息")


class DocumentInfo(BaseModel):
    """文档信息"""

    doc_id: str = Field(..., description="文档 ID")
    filename: str = Field(..., description="文件名")
    chunk_count: int = Field(default=0, description="文档块数量")
    upload_time: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="上传时间"
    )
    file_size: int = Field(default=0, description="文件大小(字节)")


class DocumentListResponse(BaseModel):
    """文档列表响应"""

    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = Field(default=0, description="文档总数")


class DocumentDeleteResponse(BaseModel):
    """文档删除响应"""

    doc_id: str = Field(..., description="文档 ID")
    message: str = Field(default="文档删除成功", description="提示信息")
