"""BM25 稀疏召回：混合检索的第二路。

为什么需要它
------------
向量检索把文本压成一个语义向量，编号、错误码、型号这类「无语义、只靠字面区分」
的串会被平滑掉 —— E102 与 E202 在向量空间里几乎重合，问 E102 可能捞回 E202 那段。
BM25 是纯字面词频匹配，恰好补上这一刀。

为什么索引文本要拼上 title + section_path
----------------------------------------
有些区分信息只在标题或章节名里，正文块里根本没有（例如「华东区」若只写在文档标题
与章节路径上）。只索正文会漏召回，所以三样拼一起建索引；但命中判定仍用原始正文，
引用与关键词校验不受影响。

为什么只在内存、不落盘
----------------------
documents / chunks / FAISS 已经三处需要同步，再加一处持久化就多一个漂移源
（见 documents.py 头注释）。BM25 索引直接从 chunks 表现场派生，用脏标记失效即可；
进程重启后第一次检索自动重建，代价可接受。

可降级
------
jieba 加载失败退回字符二元组；BM25 不可用或推理异常一律返回空列表，
让检索退回纯向量顺序，绝不中断主流程（与 rerank.py 同风格）。
"""

import re

from langchain_core.documents import Document

from app.core.logging import logger
from app.db.connection import get_conn

# jieba 懒加载一次；失败后标记不再重试，避免每次检索都卡在 import 上
_jieba = None
_jieba_ready = False


def _get_jieba():
    global _jieba, _jieba_ready
    if _jieba_ready:
        return _jieba
    _jieba_ready = True
    try:
        import jieba

        _jieba = jieba
        logger.info("jieba 分词器加载完成")
    except Exception as e:
        logger.warning("jieba 加载失败（%s），BM25 退回字符二元组分词", e)
        _jieba = None
    return _jieba


def _bigrams(text: str) -> list[str]:
    """无 jieba 时的降级分词：去掉空白后切成相邻二元组，保证中文也有重合词元。"""
    s = re.sub(r"\s+", "", text)
    if len(s) <= 1:
        return [s] if s else []
    return [s[i : i + 2] for i in range(len(s) - 1)]


def tokenize(text: str) -> list[str]:
    """统一分词入口：生产检索与评估脚本共用，保证两边口径一致。"""
    jieba = _get_jieba()
    if jieba is None:
        return _bigrams(text)
    return [t for t in jieba.lcut(text) if t.strip()]


def index_text(title: str | None, section_path: str | None, content: str) -> str:
    """BM25 索引文本 = title + section_path + content 三合一。"""
    return f"{title or ''} {section_path or ''} {content or ''}"


def doc_key(doc: Document) -> str:
    """给文档一个稳定主键，用于融合去重。

    生产文档带 chunk_id；评估文档没有，退回 title#chunk_idx，两者都能唯一定位一个块。
    """
    md = doc.metadata
    if md.get("chunk_id"):
        return str(md["chunk_id"])
    return f"{md.get('title', '')}#{md.get('chunk_idx')}"


def rrf_fuse(
    rank_lists: list[list[Document]],
    k: int,
    tiebreak_list: list[Document] | None = None,
) -> list[Document]:
    """RRF（Reciprocal Rank Fusion）融合多路召回结果。

    score(d) = Σ_lists 1 / (k + rank)，rank 从 1 开始。

    只比名次、不比分数：各路分数量纲不同（向量是距离、BM25 是词频分），
    直接加权没法调；换算成名次就天然可比。常数 k 用来压平头部，
    让「两路都靠前」胜过「单路第一」，削弱单路噪声通吃。

    tiebreak_list：RRF 只吃名次，会丢掉「谁更精确」的信息。当两路把同样的两个
    候选对调（A 路 A>B、B 路 B>A），RRF 分值完全相等，只能靠平局裁决。
    这里用 tiebreak_list 的名次兜底 —— 传入更精确的那一路（本项目的 BM25），
    字面证据更强的排前，正好补向量平滑掉编号/错误码的短板。
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for lst in rank_lists:
        for rank, doc in enumerate(lst, start=1):
            key = doc_key(doc)
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    # 平局裁决：名次越靠前值越大（取负号让排序方向一致），未出现的给最低
    tb_rank: dict[str, int] = {}
    if tiebreak_list:
        for rank, doc in enumerate(tiebreak_list, start=1):
            tb_rank[doc_key(doc)] = -rank
    _FLOOR = -10**9

    order = sorted(
        scores,
        key=lambda key: (scores[key], tb_rank.get(key, _FLOOR)),
        reverse=True,
    )
    return [docs[key] for key in order]


# ==========内存索引（从 chunks 表现场派生）==========
_index = None
_docs: list[Document] = []
_dirty = True


def mark_dirty() -> None:
    """文档增删后调用：下次检索时重建索引。"""
    global _dirty
    _dirty = True


def _load_rows():
    """从 chunks + documents 现场派生索引语料，不新增任何持久化。"""
    conn = get_conn()
    try:
        return conn.execute(
            """SELECT c.chunk_id, c.doc_id, c.chunk_idx, c.content,
                      c.page_no, c.section_path, d.title
               FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
               ORDER BY c.doc_id, c.chunk_idx"""
        ).fetchall()
    finally:
        conn.close()


def _ensure_index() -> bool:
    """按需重建索引。返回 False 表示 BM25 不可用（调用方退回纯向量）。"""
    global _index, _docs, _dirty
    if _index is not None and not _dirty:
        return True
    try:
        from rank_bm25 import BM25Okapi
    except Exception as e:
        logger.warning("rank_bm25 不可用（%s），混合检索退回纯向量", e)
        return False

    rows = _load_rows()
    if not rows:
        _index, _docs, _dirty = None, [], False
        return False

    corpus: list[list[str]] = []
    docs: list[Document] = []
    for r in rows:
        corpus.append(tokenize(index_text(r["title"], r["section_path"], r["content"])))
        docs.append(
            Document(
                page_content=r["content"] or "",
                metadata={
                    "doc_id": r["doc_id"],
                    "chunk_id": r["chunk_id"],
                    "title": r["title"],
                    "chunk_idx": r["chunk_idx"],
                    "page_no": r["page_no"],
                    "section_path": r["section_path"],
                },
            )
        )
    try:
        _index = BM25Okapi(corpus)
    except Exception as e:
        logger.warning("BM25 索引构建失败（%s），混合检索退回纯向量", e)
        return False
    _docs = docs
    _dirty = False
    logger.info("BM25 索引重建完成 文本块=%d", len(docs))
    return True


def search(query: str, top_k: int, request_id: str = "-") -> list[Document]:
    """BM25 召回 top_k 条。任何异常都返回空列表，调用方据此退回纯向量。"""
    if top_k <= 0 or not _ensure_index():
        return []
    try:
        scores = _index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        docs = [_docs[i] for i in order]
        logger.info(
            "[%s] BM25 召回 候选=%d 保留=%d 首条得分=%.3f",
            request_id,
            len(scores),
            len(docs),
            float(scores[order[0]]) if order else 0.0,
        )
        return docs
    except Exception as e:
        logger.warning("[%s] BM25 召回失败（%s），退回纯向量", request_id, e)
        return []