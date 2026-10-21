# -*- coding: utf-8 -*-
"""把 eval/ 下所有 report*.json 汇总成 Markdown 对比表，打印到 stdout。

只取 k=1 和 k=3 的指标（Hit / MRR / SecHit / SecMRR），
按「语料组」分组排序，方便直接贴进 README。
"""
import json
from pathlib import Path

EVAL_DIR = Path(r"C:\Users\Administrator\PycharmProjects\AIChat\eval")


def row_for(report_path: Path) -> dict | None:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {s["k"]: s for s in data.get("summary", [])}
    if not summary:
        return None
    k1 = summary.get(1, {})
    k3 = summary.get(3, {})
    total = next((s["total"] for s in data.get("summary", [])), 0)
    return {
        "file": report_path.name,
        "hybrid": data.get("hybrid", False),
        "rerank": data.get("rerank", False),
        "total": total,
        "hit1": k1.get("hit_rate"),
        "mrr1": k1.get("mrr"),
        "sechit1": k1.get("section_hit_rate"),
        "secmrr1": k1.get("section_mrr"),
        "hit3": k3.get("hit_rate"),
        "sechit3": k3.get("section_hit_rate"),
    }


def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def config_label(hybrid: bool, rerank: bool) -> str:
    if not hybrid and not rerank:
        return "纯向量检索"
    if hybrid and not rerank:
        return "向量 + BM25 混合"
    if not hybrid and rerank:
        return "向量 + 重排"
    return "向量 + BM25 + 重排（完整）"


def main() -> None:
    rows = []
    for p in sorted(EVAL_DIR.glob("report*.json")):
        r = row_for(p)
        if r:
            rows.append(r)

    print("| 文件 | hybrid | rerank | 题数 | Hit@1 | SecHit@1 | MRR@1 | Hit@3 | SecHit@3 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['file']} | {'是' if r['hybrid'] else '否'} | {'是' if r['rerank'] else '否'} | "
            f"{r['total']} | {fmt(r['hit1'])} | {fmt(r['sechit1'])} | {fmt(r['mrr1'])} | "
            f"{fmt(r['hit3'])} | {fmt(r['sechit3'])} |"
        )


if __name__ == "__main__":
    main()
