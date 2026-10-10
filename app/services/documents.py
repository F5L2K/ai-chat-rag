"""文档入库与查询。

三处必须保持一致，增删都要同时动，否则会漂移：
  1. documents 表 —— 文档级元数据（去重、权限、状态）
  2. chunks 表    —— 分块明细（全文 + 页码 + 章节），BM25 与重建索引都靠它
  3. FAISS 索引   —— 向量
"""

import hashlib
import uuid

from langchain_core.documents import Document

from app.core.logging import logger
from app.db.connection import get_conn
from app.services.chunking import build_chunks
from app.services.parsing import ParseResult, parse_text
from app.services.vector_store import get_vector_store, save_vector_store


def content_hash(content: str) -> str:
    """按内容算指纹，用于识别重复文档。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def find_by_hash(hash_value: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT doc_id, title, chunk_count FROM documents WHERE content_hash = ?",
        (hash_value,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def index_document(
    title: str,
    parsed: ParseResult,
    source_path: str | None = None,
    file_hash: str | None = None,
) -> dict:
    """把解析结果切块，写入 documents / chunks / FAISS。

    内容完全相同的文档视为重复，直接返回已有记录，不重复入库。
    解析不出任何文字时（扫描件）只记录状态，不进向量库。
    """
    full_text = "\n".join(b.text for b in parsed.blocks)
    hash_value = file_hash or content_hash(full_text)

    existing = find_by_hash(hash_value)
    if existing:
        logger.info("重复文档，跳过入库 title=%s doc_id=%s", title, existing["doc_id"])
        return {**existing, "duplicated": True, "needs_ocr": False, "status": "ready"}

    chunks = build_chunks(parsed.blocks)
    doc_id = str(uuid.uuid4())
    status = "needs_ocr" if (parsed.needs_ocr or not chunks) else "ready"

    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO documents
               (doc_id, title, source_type, source_path, content_hash,
                page_count, chunk_count, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                title,
                parsed.source_type,
                source_path,
                hash_value,
                parsed.page_count,
                len(chunks),
                status,
            ),
        )
        if chunks:
            conn.executemany(
                """INSERT INTO chunks
                   (chunk_id, doc_id, chunk_idx, content,
                    page_no, section_path, char_start, char_end)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        f"{doc_id}:{i}",
                        doc_id,
                        i,
                        c.text,
                        c.page_no,
                        c.section_path,
                        c.char_start,
                        c.char_end,
                    )
                    for i, c in enumerate(chunks)
                ],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if chunks:
        chunk_ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        # 元数据一次写全：第 3 刀做引用、第 4 刀做权限过滤，都直接读这些字段
        docs = [
            Document(
                page_content=c.text,
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": chunk_ids[i],
                    "title": title,
                    "chunk_idx": i,
                    "page_no": c.page_no,
                    "section_path": c.section_path,
                    "tenant_id": "default",
                    "dept_id": None,
                    "acl_tags": [],
                },
            )
            for i, c in enumerate(chunks)
        ]
        get_vector_store().add_documents(docs, ids=chunk_ids)
        save_vector_store()

    logger.info(
        "文档已入库 title=%s doc_id=%s 切块=%d 页数=%s 状态=%s",
        title, doc_id, len(chunks), parsed.page_count, status,
    )
    return {
        "doc_id": doc_id,
        "title": title,
        "chunk_count": len(chunks),
        "page_count": parsed.page_count,
        "needs_ocr": parsed.needs_ocr,
        "status": status,
        "duplicated": False,
    }


def add_text_document(title: str, content: str) -> dict:
    """纯文本入库（JSON 上传走这条路）。"""
    return index_document(title, parse_text(content))


def list_documents() -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT doc_id, title, source_type, source_path, version,
               page_count, chunk_count, status, created_at
        FROM documents
        ORDER BY created_at DESC, rowid DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    """文档详情：元数据 + 全部分块（带页码/章节，供前端预览与排查）。"""
    conn = get_conn()
    try:
        row = conn.execute(
            """SELECT doc_id, title, source_type, source_path, version,
                      page_count, chunk_count, status, created_at
               FROM documents WHERE doc_id = ?""",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        doc = dict(row)
        chunks = conn.execute(
            """SELECT chunk_idx, page_no, section_path, content
               FROM chunks WHERE doc_id = ? ORDER BY chunk_idx""",
            (doc_id,),
        ).fetchall()
        doc["chunks"] = [dict(c) for c in chunks]
        return doc
    finally:
        conn.close()


def delete_document(doc_id: str) -> bool:
    """删除文档：documents / chunks / FAISS 三处一起清。

    只删表不清向量，检索时就会召回已经不存在的文档（漂移），
    所以这三步必须一起做，且先落表再落向量，任一步失败都不会留下半截状态。
    """
    conn = get_conn()
    try:
        exists = conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if exists is None:
            return False

        chunk_ids = [
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchall()
        ]
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if chunk_ids:
        store = get_vector_store()
        store.delete(chunk_ids)
        save_vector_store()

    logger.info("文档已删除 doc_id=%s 切块=%d", doc_id, len(chunk_ids))
    return True
