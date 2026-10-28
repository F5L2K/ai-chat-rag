"""知识库接口。

两个上传入口：
  POST /api/knowledge/upload       JSON 传纯文本
  POST /api/knowledge/upload-file  multipart 传文件（PDF / Word / Markdown / txt / HTML）
"""

import io
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.core.config import CHAT_MODEL
from app.core.logging import logger
from app.models.schemas import DocUploadRequest
from app.services.documents import (
    add_text_document,
    delete_document,
    get_document,
    index_document,
    list_documents,
    update_document,
)
from app.services.llm import client
from app.services.parsing import parse_document

router = APIRouter()


@router.post("/upload")
async def upload_doc(req: DocUploadRequest, request: Request):
    """上传一篇纯文本文档：切块 → 向量化 → 入库。"""
    request_id = getattr(request.state, "request_id", "-")
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail={
            "code": "invalid_doc",
            "message": "标题和内容都不能为空",
        })
    started = time.perf_counter()
    try:
        result = add_text_document(req.title, req.content)
    except Exception as e:
        # embedding 失败（如 key 无效、网络问题）
        logger.exception("[%s] 文档向量化失败 title=%s", request_id, req.title)
        raise HTTPException(status_code=500, detail={
            "code": "embedding_failed",
            "message": f"文档向量化失败：{str(e)}",
        })
    logger.info(
        "[%s] 文本入库完成 title=%s 切块=%d 重复=%s 耗时=%.0fms",
        request_id, req.title, result["chunk_count"], result["duplicated"],
        (time.perf_counter() - started) * 1000,
    )
    return result


@router.post("/upload-file")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
):
    """上传文件：解析 → 切块 → 向量化 → 入库。

    解析出的每一块都带页码（PDF）和章节路径（三种格式都有），
    这两个字段是后续「带引用回答」的原料。
    """
    request_id = getattr(request.state, "request_id", "-")
    filename = file.filename or "unnamed"
    started = time.perf_counter()

    # 先整块读进内存再包成 BytesIO：python-docx 底层的 zipfile 需要可 seek 的流，
    # 而 FastAPI 的 SpooledTemporaryFile 不满足，直接传会解析失败
    data = await file.read()

    try:
        parsed = parse_document(filename, io.BytesIO(data))
    except ValueError as e:
        # 类型不支持、老 .doc 等：属于用户输入问题，返回 400 并给出可操作提示
        raise HTTPException(status_code=400, detail={
            "code": "unsupported_file",
            "message": str(e),
        })
    except Exception as e:
        logger.exception("[%s] 文件解析失败 file=%s", request_id, filename)
        raise HTTPException(status_code=400, detail={
            "code": "parse_failed",
            "message": f"文件解析失败：{e}",
        })

    doc_title = (title or Path(filename).stem).strip() or filename

    try:
        result = index_document(doc_title, parsed, source_path=filename)
    except Exception as e:
        logger.exception("[%s] 文档入库失败 file=%s", request_id, filename)
        raise HTTPException(status_code=500, detail={
            "code": "index_failed",
            "message": f"文档入库失败：{e}",
        })

    logger.info(
        "[%s] 文件上传完成 file=%s 切块=%d 页数=%s 耗时=%.0fms",
        request_id, filename, result["chunk_count"], result["page_count"],
        (time.perf_counter() - started) * 1000,
    )
    return result


@router.get("")
async def list_docs():
    """列出知识库中所有文档。"""
    return {"docs": list_documents()}


@router.get("/{doc_id}")
async def get_doc(doc_id: str):
    """文档详情：元数据 + 全部分块（含页码/章节），供前端预览。"""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail={
            "code": "doc_not_found",
            "message": "文档不存在",
        })
    return doc


@router.delete("/{doc_id}")
async def remove_doc(doc_id: str, request: Request):
    """删除文档：表与向量库一起清，避免检索召回已删文档。"""
    request_id = getattr(request.state, "request_id", "-")
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={
            "code": "doc_not_found",
            "message": "文档不存在",
        })
    logger.info("[%s] 文档已删除 doc_id=%s", request_id, doc_id)
    return {"deleted": 1, "doc_id": doc_id}


@router.post("/{doc_id}/reupload")
async def reupload_file(
    doc_id: str,
    request: Request,
    file: UploadFile = File(...),
):
    """用新版本文件替换已有文档：旧 chunks 与向量同步失效，version+1。

    对应知识更新流程：制度改版后不用删了重传，直接传新版本上来。
    内容没变（content_hash 相同）时返回 unchanged=true，不动数据。
    """
    request_id = getattr(request.state, "request_id", "-")
    filename = file.filename or "unnamed"
    started = time.perf_counter()
    data = await file.read()

    try:
        parsed = parse_document(filename, io.BytesIO(data))
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "code": "unsupported_file",
            "message": str(e),
        })
    except Exception as e:
        logger.exception("[%s] 更新文件解析失败 file=%s", request_id, filename)
        raise HTTPException(status_code=400, detail={
            "code": "parse_failed",
            "message": f"文件解析失败：{e}",
        })

    result = update_document(doc_id, parsed, source_path=filename)
    if result is None:
        raise HTTPException(status_code=404, detail={
            "code": "doc_not_found",
            "message": "文档不存在",
        })
    logger.info(
        "[%s] 文件更新完成 file=%s updated=%s 耗时=%.0fms",
        request_id, filename, result.get("updated"),
        (time.perf_counter() - started) * 1000,
    )
    return result


@router.post("/{doc_id}/summarize")
async def summarize_doc(doc_id: str, request: Request):
    """对知识库里已有文档生成摘要：从 chunks 拼全文，交给 LLM 概括。

    属于「结合业务场景开发 AI 应用」里的文档摘要能力。
    全文截断到 _SUMMARY_MAX_CHARS 防止超出模型上下文。
    """
    request_id = getattr(request.state, "request_id", "-")
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail={
            "code": "doc_not_found",
            "message": "文档不存在",
        })

    full_text = "\n".join(c["content"] for c in doc.get("chunks", []))
    if not full_text.strip():
        raise HTTPException(status_code=400, detail={
            "code": "empty_doc",
            "message": "文档没有可摘要的文字内容",
        })

    # 摘要不需要全文，截断到 6000 字以内足够覆盖要点
    _SUMMARY_MAX_CHARS = 6000
    trimmed = full_text[:_SUMMARY_MAX_CHARS]
    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是企业文档摘要助手。请用 200 字以内概括下面文档的核心内容，"
                               "分 3-5 个要点输出，不要输出与文档无关的内容。",
                },
                {"role": "user", "content": trimmed},
            ],
            stream=False,
        )
    except Exception as e:
        logger.exception("[%s] 摘要生成失败 doc_id=%s", request_id, doc_id)
        raise HTTPException(status_code=500, detail={
            "code": "summary_failed",
            "message": f"摘要生成失败：{e}",
        })

    summary = resp.choices[0].message.content or ""
    logger.info(
        "[%s] 摘要完成 doc_id=%s 原文=%d字 摘要=%d字 耗时=%.0fms",
        request_id, doc_id, len(full_text), len(summary),
        (time.perf_counter() - started) * 1000,
    )
    return {
        "doc_id": doc_id,
        "title": doc["title"],
        "summary": summary,
        "source_chars": len(full_text),
    }
