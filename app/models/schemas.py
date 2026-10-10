"""接口的数据契约。请求进来先过这里，字段不对直接返回 422。"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class DocUploadRequest(BaseModel):
    title: str
    content: str
