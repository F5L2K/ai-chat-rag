"""检索评估：把「知识库检索准不准」变成可对比的数字。

三个设计决定：
1. 独立索引 —— 在内存里另建一个 FAISS，不读写 chat.db / faiss_store，
   所以反复跑评估不会污染真实知识库，也不需要清场。
2. 复用生产代码 —— 解析、切分、embedding 全部走 app/services 里的同一套实现，
   评估的就是线上那套逻辑，而不是另写一个近似版本。
3. 只评检索层 —— 模型不参与打分。这样「回答不对」时能分清是
   「资料没找对」还是「找对了但模型没说好」，两者的修法完全不同。

指标说明：
  Hit@K     —— 前 K 个结果里有没有正确来源（命中率）
  MRR@K     —— 正确来源排名的倒数平均，排得越靠前分越高
  SecHit@K  —— 命中的块还落在期望章节里的比例（引用定位准不准）
  SecMRR@K  —— 章节命中排名的倒数平均
  SecHit@K ⊆ Hit@K，两者之差 = 「文档找对了，但引用章节没对上」
  这里不报 Recall：每题只有一个正确来源，Recall 会和 Hit 完全重复，属于噪声。

  章节级指标只在标注集带 expect_section 字段时才计算；旧数据集自动降级。

用法：
    python eval/run_eval.py                    # 默认语料 corpus/，用生产默认 top_k
    python eval/run_eval.py --top-k 1 3 5      # 扫描多个 K，看加宽召回有没有收益
    python eval/run_eval.py --verbose          # 打印逐题命中排名，定位坏例

    # 开启重排序，对比「向量排序 vs 交叉精排」（复用生产同一套 rerank 逻辑）
    python eval/run_eval.py --rerank --report report_rerank.json

    # 开启混合检索（向量 + BM25，RRF 融合），可再叠加 --rerank 看完整三阶段
    python eval/run_eval.py --hybrid --report report_hybrid.json
    python eval/run_eval.py --hybrid --rerank --report report_hybrid_rerank.json

    # 换一套语料与标注集（例如用真实文档）
    python eval/run_eval.py --corpus corpus_real \
        --dataset dataset_real.json --report report_real.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import RAG_BM25_TOP_K, RAG_RECALL_K, RAG_RRF_K, RAG_TOP_K
from app.services.bm25 import index_text, rrf_fuse, tokenize
from app.services.chunking import build_chunks
from app.services.embeddings import get_embeddings
from app.services.parsing import SUPPORTED_SUFFIXES, parse_document
from app.services.rerank import rerank_documents

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_DIR = Path(__file__).resolve().parent


def load_corpus(corpus_dir: Path) -> tuple[list[Document], list[dict]]:
    """按生产逻辑切块，把语料收集成待索引的 Document 列表。"""
    files = [
        p
        for p in sorted(corpus_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        raise SystemExit(f"语料目录为空：{corpus_dir}")

    docs: list[Document] = []
    stats: list[dict] = []
    for path in files:
        parsed = parse_document(path.name, str(path))
        chunks = build_chunks(parsed.blocks)
        stats.append({"title": path.stem, "chunks": len(chunks)})
        for i, c in enumerate(chunks):
            docs.append(
                Document(
                    page_content=c.text,
                    metadata={
                        "title": path.stem,
                        "chunk_idx": i,
                        "section_path": c.section_path,
                        "page_no": c.page_no,
                    },
                )
            )
    return docs, stats


def hit_matches(doc: Document, item: dict) -> bool:
    """命中判定：来源文档对得上，且关键词全部出现。

    只看来源文档不够 —— 同一篇文档里可能有多个块，
    关键词保证命中的是「含答案的那个块」，而不是碰巧同一篇的其它块。
    """
    if doc.metadata.get("title") != item["doc"]:
        return False
    return all(kw in doc.page_content for kw in item.get("must_contain", []))


def section_matches(doc: Document, item: dict) -> bool:
    """章节命中：在「命中」的基础上，块还要落在期望章节里。

    衡量的是「引用定位准不准」—— 文档找对了还不够，
    引用指向的章节也必须正确，否则用户按引用回查会找错地方。
    """
    if not hit_matches(doc, item):
        return False
    expect = item.get("expect_section")
    return bool(expect) and expect in (doc.metadata.get("section_path") or "")


def rank_of_first(hits: list[Document], item: dict, matcher) -> int | None:
    for rank, doc in enumerate(hits, start=1):
        if matcher(doc, item):
            return rank
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库检索评估")
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=None,
        help=f"要评估的 K 值列表，可给多个（默认取生产配置 RAG_TOP_K={RAG_TOP_K}）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印逐题命中排名")
    parser.add_argument(
        "--corpus", default="corpus", help="语料目录名，相对 eval/（默认 corpus）"
    )
    parser.add_argument(
        "--dataset",
        default="dataset.json",
        help="标注集文件名，相对 eval/（默认 dataset.json）",
    )
    parser.add_argument(
        "--report",
        default="report.json",
        help="报告输出文件名，相对 eval/（默认 report.json）",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="启用 Cross-encoder 重排序（先按 recall-k 召回，再精排到 max_k）",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help=f"启用混合检索（向量 + BM25，RRF 融合，BM25 取 {RAG_BM25_TOP_K}、k={RAG_RRF_K}）",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=None,
        help=f"重排序前的召回池大小（默认 RAG_RECALL_K={RAG_RECALL_K}）",
    )
    args = parser.parse_args()

    ks = sorted(set(args.top_k or [RAG_TOP_K]))
    max_k = max(ks)

    corpus_dir = EVAL_DIR / args.corpus
    dataset_path = EVAL_DIR / args.dataset
    report_path = EVAL_DIR / args.report

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))["questions"]
    # 标注集带 expect_section 时才算章节级指标；旧数据集自动降级，不受影响
    has_section = any(item.get("expect_section") for item in dataset)

    print(f"语料目录：{corpus_dir.name}/")
    print(f"标注集：{dataset_path.name}")
    print("正在构建评估索引（首次运行需要加载 embedding 模型，约 10 秒）...")
    docs, stats = load_corpus(corpus_dir)
    store = FAISS.from_documents(docs, get_embeddings())
    print(f"语料：{len(stats)} 篇文档 / {len(docs)} 个文本块")
    for s in stats:
        print(f"  - {s['title']}：{s['chunks']} 块")
    print(f"问题：{len(dataset)} 条")
    # 召回池：开启重排序时多召回候选，给精排留出腾挪空间
    recall_k = max(max_k, args.recall_k or RAG_RECALL_K)

    # 混合检索：用同一批 docs 建内存 BM25 索引（与生产同款 tokenize / index_text）
    bm25 = None
    if args.hybrid:
        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(
                [
                    tokenize(
                        index_text(
                            d.metadata.get("title"),
                            d.metadata.get("section_path"),
                            d.page_content,
                        )
                    )
                    for d in docs
                ]
            )
        except Exception as e:
            print(f"BM25 不可用（{e}），本次退回纯向量")
            bm25 = None

    stages = f"稠密：向量召回 {min(recall_k, len(docs))}"
    if args.rerank:
        stages += " → 精排"
    if bm25 is not None:
        stages += f" ∥ 稀疏：BM25 召回 {RAG_BM25_TOP_K} → RRF 融合"
    print(f"检索模式：{stages} → top {max_k}\n")

    rows = []
    for item in dataset:
        # 稠密路：向量召回，再整批精排出顺序（与生产同款：融合放在精排之后）
        dense = store.similarity_search(
            item["question"], k=min(recall_k, len(docs))
        )
        if args.rerank:
            dense = rerank_documents(
                item["question"], dense, min(recall_k, len(docs)), "eval"
            )
        # 稀疏路：BM25 召回，与稠密路 RRF 融合（BM25 名次作平局裁决）
        if bm25 is not None:
            scores = bm25.get_scores(tokenize(item["question"]))
            order = sorted(
                range(len(docs)), key=lambda i: scores[i], reverse=True
            )[:RAG_BM25_TOP_K]
            bm25_hits = [docs[i] for i in order]
            hits = rrf_fuse([dense, bm25_hits], RAG_RRF_K, tiebreak_list=bm25_hits)[
                :max_k
            ]
        else:
            hits = dense[:max_k]
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "doc": item["doc"],
                "expect_section": item.get("expect_section"),
                "rank": rank_of_first(hits, item, hit_matches),
                "section_rank": (
                    rank_of_first(hits, item, section_matches) if has_section else None
                ),
            }
        )

    total = len(rows)
    summary = []
    for k in ks:
        hit = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k)
        rr = sum(1.0 / r["rank"] for r in rows if r["rank"] is not None and r["rank"] <= k)
        row = {
            "k": k,
            "hit_rate": hit / total,
            "mrr": rr / total,
            "hits": hit,
            "total": total,
        }
        if has_section:
            s_hit = sum(
                1 for r in rows if r["section_rank"] is not None and r["section_rank"] <= k
            )
            s_rr = sum(
                1.0 / r["section_rank"]
                for r in rows
                if r["section_rank"] is not None and r["section_rank"] <= k
            )
            row["section_hit_rate"] = s_hit / total
            row["section_mrr"] = s_rr / total
            row["section_hits"] = s_hit
        summary.append(row)

    print("=== 指标 ===")
    if has_section:
        print(
            f"{'K':>3}  {'Hit@K':>8}  {'MRR@K':>8}  {'SecHit@K':>9}  {'SecMRR@K':>9}   命中/总数"
        )
        for s in summary:
            print(
                f"{s['k']:>3}  {s['hit_rate'] * 100:>7.1f}%  {s['mrr']:>8.3f}  "
                f"{s['section_hit_rate'] * 100:>8.1f}%  {s['section_mrr']:>9.3f}   "
                f"{s['hits']}/{s['total']}"
            )
    else:
        print(f"{'K':>3}  {'Hit@K':>8}  {'MRR@K':>8}   命中/总数")
        for s in summary:
            print(
                f"{s['k']:>3}  {s['hit_rate'] * 100:>7.1f}%  {s['mrr']:>8.3f}   "
                f"{s['hits']}/{s['total']}"
            )

    misses = [r for r in rows if r["rank"] is None]
    if misses:
        print(f"\n=== 完全没召回的问题（{len(misses)} 条）===")
        for r in misses:
            print(f"  [{r['id']}] {r['question']}  → 期望：{r['doc']}")

    if has_section:
        # 文档找对了、但引用没落到正确章节的题：章节级指标要盯的就是这批
        gaps = [
            r
            for r in rows
            if r["expect_section"] and r["rank"] is not None and r["section_rank"] is None
        ]
        if gaps:
            print(f"\n=== 文档命中但章节定位不对（{len(gaps)} 条）===")
            for r in gaps:
                print(f"  [{r['id']}] 期望章节：{r['expect_section']}  ← {r['question']}")

    if args.verbose:
        print("\n=== 逐题命中排名 ===")
        for r in rows:
            mark = f"第 {r['rank']} 名" if r["rank"] else "未召回"
            if has_section and r["expect_section"]:
                s_mark = (
                    f"章节第 {r['section_rank']} 名" if r["section_rank"] else "章节未命中"
                )
                print(f"  [{r['id']}] {mark:<8} {s_mark:<12} {r['question']}")
            else:
                print(f"  [{r['id']}] {mark:<8} {r['question']}")

    report_path.write_text(
        json.dumps(
            {
                "rerank": args.rerank,
                "hybrid": bm25 is not None,
                "bm25_top_k": RAG_BM25_TOP_K if bm25 is not None else None,
                "rrf_k": RAG_RRF_K if bm25 is not None else None,
                "recall_k": recall_k if (args.rerank or bm25 is not None) else None,
                "corpus": stats,
                "summary": summary,
                "questions": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告已写入：{report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
