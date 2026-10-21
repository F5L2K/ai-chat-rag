"""重排序：两阶段检索的第二阶段。

第一阶段（FAISS 向量召回）负责「找得全」——双塔模型把问题和文档各自编码成向量，
快，但两者之间没有词级交互，细粒度匹配容易排偏。
第二阶段（Cross-encoder 精排）负责「排得准」——把 (问题, 文档) 配对送进模型打分，
能捕捉词级交互，精度更高但慢，所以只在「少量候选」上做。

流程：FAISS 召回 RAG_RECALL_K 个候选 → Cross-encoder 精排出 RAG_TOP_K 条。

可降级：本模型与 embedding 同源（bge 系列），首次需下载；若加载或推理失败，
自动退回原始向量顺序，保证检索链路不因重排序而中断。
"""

import time

from langchain_core.documents import Document

from app.core.config import RAG_RERANK_MODEL
from app.core.logging import logger

_instance = None
# 加载失败标记：模型下载失败后不再反复重试，避免每次请求都卡在网络超时上
_load_failed = False


def _get_model():
    """懒加载 Cross-encoder。返回 None 表示不可用（调用方据此降级）。"""
    global _instance, _load_failed
    if _instance is not None:
        return _instance
    if _load_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder

        logger.info("正在加载重排序模型：%s", RAG_RERANK_MODEL)
        started = time.perf_counter()
        _instance = CrossEncoder(RAG_RERANK_MODEL)
        logger.info(
            "重排序模型加载完成 耗时=%.0fms",
            (time.perf_counter() - started) * 1000,
        )
    except Exception as e:
        _load_failed = True
        logger.warning("重排序模型加载失败（%s），后续检索退回向量顺序", e)
    return _instance


def rerank_documents(
    query: str, docs: list[Document], top_k: int, request_id: str = "-"
) -> list[Document]:
    """对候选文档精排，返回前 top_k 条。

    任何异常都退化为「不重排」，保证调用方始终能拿到结果。
    """
    if len(docs) <= 1:
        return docs[:top_k]

    model = _get_model()
    if model is None:
        return docs[:top_k]

    try:
        pairs = [[query, d.page_content] for d in docs]
        scores = model.predict(pairs)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        ranked = [docs[i] for i in order]
        logger.info(
            "[%s] 重排序完成 候选=%d 保留=%d 首条得分=%.3f",
            request_id, len(docs), min(top_k, len(ranked)), float(scores[order[0]]),
        )
        return ranked[:top_k]
    except Exception as e:
        logger.warning("[%s] 重排序推理失败（%s），退回向量顺序", request_id, e)
        return docs[:top_k]