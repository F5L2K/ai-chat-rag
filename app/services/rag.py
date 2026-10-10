"""检索与 prompt 组装。"""

from app.core.config import RAG_TOP_K, SYSTEM_PROMPT
from app.core.logging import logger, preview
from app.services.vector_store import get_vector_store


def retrieve_relevant(query: str, request_id: str = "-") -> list:
    """根据问题检索最相关的文本片段，返回字符串列表。向量库为空时返回空列表。

    这里必须走 get_vector_store() 做懒加载，不能直接判断全局变量是否为 None：
    全局变量只在「上传文档」时被赋值，后端一重启它就是 None，
    检索会永远返回空列表 —— 模型拿不到资料却照常作答，属于静默降级，不报任何错。
    """
    store = get_vector_store()
    total = store.index.ntotal
    if total == 0:
        logger.warning("[%s] 向量库为空，本次检索跳过（知识库里还没有文档）", request_id)
        return []
    docs = store.similarity_search(query, k=RAG_TOP_K)
    logger.info(
        "[%s] 检索完成 向量库总量=%d 命中=%d 问题=%s",
        request_id, total, len(docs), preview(query, 40),
    )
    return [doc.page_content for doc in docs]


def build_rag_prompt(query: str, request_id: str = "-") -> str:
    """把检索到的资料拼进 system prompt，让模型根据资料回答。"""
    chunks = retrieve_relevant(query, request_id)
    if not chunks:
        # 不静默：这条日志是排查「回答没依据」时的第一现场
        logger.warning("[%s] 未检索到资料，本次回答不携带知识库上下文", request_id)
        return SYSTEM_PROMPT
    # 把相关片段拼起来，用明确的分隔标记
    context = "\n\n---\n\n".join(
        f"资料片段 {i + 1}：\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"以下是可能相关的参考资料，请根据资料回答用户问题。"
        f"如果资料里没有答案，就如实说不知道，不要编造。\n\n"
        f"===== 参考资料 =====\n"
        f"{context}\n"
        f"===== 参考资料结束 ====="
    )
