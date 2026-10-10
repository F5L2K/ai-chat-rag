"""FAISS 向量库：懒加载 + 落盘持久化。"""

import os
import shutil

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from app.core.config import FAISS_INDEX_PATH
from app.core.logging import logger
from app.services.embeddings import get_embeddings


def _init_store() -> FAISS:
    """加载或创建向量库。

    本地已有索引文件就加载；没有就新建空的。
    旧索引加载失败（比如换了 embedding 模型导致维度不匹配）就删掉重建。
    """
    emb = get_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            store = FAISS.load_local(
                FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True
            )
            logger.info("已从磁盘恢复向量库：%d 个向量", store.index.ntotal)
            return store
        except Exception as e:
            logger.warning("旧向量库加载失败（%s），将删除重建", e)
            shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
    # 新建空库：先拿一句样例文本测一下维度，再用这个维度建索引
    sample = emb.embed_query("init")
    dimension = len(sample)
    index = faiss.IndexFlatL2(dimension)
    logger.info("新建空向量库 dimension=%d", dimension)
    return FAISS(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


_store: FAISS | None = None


def get_vector_store() -> FAISS:
    """获取向量库实例，需要时才初始化。"""
    global _store
    if _store is None:
        _store = _init_store()
    return _store


def save_vector_store():
    """把当前向量库保存到磁盘。每次改动文档后调用。"""
    get_vector_store().save_local(FAISS_INDEX_PATH)
