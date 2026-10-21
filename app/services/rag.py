"""检索与 prompt 组装。

检索是「稠密 + 稀疏」双路，融合后取 top-K：
  1. 稠密路 —— FAISS 向量召回 RAG_RECALL_K 个（管语义），再用 Cross-encoder
     整批精排出顺序（RAG_RERANK_ENABLED=0 时跳过精排）
  2. 稀疏路 —— BM25 召回 RAG_BM25_TOP_K 个（管字面：E102 / AUTH-1003 这类
     向量平滑掉的编号，只有字面匹配能分开）
  3. RRF 融合两路名次，平局以 BM25 名次裁决，取 RAG_TOP_K 条

为什么融合放在精排「之后」：Cross-encoder 是 pointwise 打分，对每个 (问题, 块)
独立评分，候选集合怎么变都不影响其余块的相对次序 —— 融合若放在精排之前，精排会
把融合出来的顺序整个丢掉，等于没融合（实测：融合前置在对抗语料上与纯精排逐题一致）。
所以先让稠密路精排出顺序，再让 BM25 的字面证据去校正精排对编号类问题的失误。

可分别关闭：RAG_HYBRID_ENABLED=0 退回单路稠密检索；RAG_RERANK_ENABLED=0 稠密路不精排。

检索结果带 metadata（title / page_no / section_path），
build_rag_prompt 返回 (prompt, citations) 元组：
  - prompt 里用编号引用资料，并指示模型用 [1] [2] 标注来源
  - citations 是结构化列表，供 chat.py 在流末尾推给前端展示
"""

from langchain_core.documents import Document

from app.core.config import (
    RAG_BM25_TOP_K,
    RAG_HYBRID_ENABLED,
    RAG_RECALL_K,
    RAG_RERANK_ENABLED,
    RAG_RRF_K,
    RAG_TOP_K,
    SYSTEM_PROMPT,
)
from app.core.logging import logger, preview
from app.services.bm25 import rrf_fuse
from app.services.bm25 import search as bm25_search
from app.services.rerank import rerank_documents
from app.services.vector_store import get_vector_store


def retrieve_relevant(query: str, request_id: str = "-") -> list[Document]:
    """稠密 + 稀疏双路检索，返回最相关的 RAG_TOP_K 个片段（含 metadata）。

    这里必须走 get_vector_store() 做懒加载，不能直接判断全局变量是否为 None：
    全局变量只在「上传文档」时被赋值，后端一重启它就是 None，
    检索会永远返回空列表 —— 模型拿不到资料却照常作答，属于静默降级，不报任何错。
    """
    store = get_vector_store()
    total = store.index.ntotal
    if total == 0:
        logger.warning("[%s] 向量库为空，本次检索跳过（知识库里还没有文档）", request_id)
        return []

    # 稠密路：向量召回。候选数取 min(召回池, 库内总量)，并保证不少于 RAG_TOP_K
    recall_k = max(RAG_TOP_K, min(RAG_RECALL_K, total))
    dense = store.similarity_search(query, k=recall_k)

    # 稠密路精排：整批重排出顺序（多排几档，给融合留腾挪空间；可关闭/可降级）
    if RAG_RERANK_ENABLED:
        dense = rerank_documents(query, dense, recall_k, request_id)

    # 稀疏路：BM25 字面召回（可关闭/可降级，不可用时返回空）
    bm25_hits = (
        bm25_search(query, RAG_BM25_TOP_K, request_id) if RAG_HYBRID_ENABLED else []
    )

    # 融合：BM25 作为平局裁决方（字面证据精确），最后取 RAG_TOP_K 条
    if bm25_hits:
        docs = rrf_fuse([dense, bm25_hits], RAG_RRF_K, tiebreak_list=bm25_hits)[
            :RAG_TOP_K
        ]
    else:
        docs = dense[:RAG_TOP_K]

    logger.info(
        "[%s] 检索完成 向量库总量=%d 稠密=%d BM25=%d 混合=%s 重排=%s 命中=%d 问题=%s",
        request_id,
        total,
        len(dense),
        len(bm25_hits),
        "开" if RAG_HYBRID_ENABLED else "关",
        "开" if RAG_RERANK_ENABLED else "关",
        len(docs),
        preview(query, 40),
    )
    return docs


def build_rag_prompt(
    query: str, request_id: str = "-"
) -> tuple[str, list[dict]]:
    """把检索到的资料拼进 system prompt，让模型根据资料回答。

    返回 (prompt, citations)：
      - prompt 里每条资料带编号，并指示模型用 [1] [2] 标注引用来源
      - citations 是 [{ref, title, section_path, page_no}] 列表，
        chat.py 在流末尾推给前端，前端展示「📎 引用来源」
    """
    docs = retrieve_relevant(query, request_id)
    if not docs:
        # 不静默：这条日志是排查「回答没依据」时的第一现场
        logger.warning("[%s] 未检索到资料，本次回答不携带知识库上下文", request_id)
        return SYSTEM_PROMPT, []

    # 每条资料带编号拼进 prompt
    context_parts = []
    citations: list[dict] = []
    for i, doc in enumerate(docs):
        context_parts.append(f"资料片段 {i + 1}：\n{doc.page_content}")
        meta = doc.metadata
        citations.append({
            "ref": i + 1,
            "title": meta.get("title", ""),
            "section_path": meta.get("section_path"),
            "page_no": meta.get("page_no"),
        })

    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"以下是可能相关的参考资料，请根据资料回答用户问题。"
        f"如果资料里没有答案，就如实说不知道，不要编造。\n"
        f"回答时在关键信息后用 [1]、[2] 等标注引用来源编号"
        f"（编号对应下方资料片段前的序号）。\n\n"
        f"===== 参考资料 =====\n"
        f"{context}\n"
        f"===== 参考资料结束 ====="
    )
    return prompt, citations
