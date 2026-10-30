# -*- coding: utf-8 -*-
"""导出医学语料的章节索引（section_path → 块预览），供标注集选题定位。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.parsing import parse_document  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus_medical"
OUT = Path(__file__).resolve().parent / "_medical_index.json"


def main() -> None:
    result = {}
    for f in sorted(CORPUS.glob("*.md")):
        res = parse_document(f.name, str(f))
        by_sec: dict[str, list] = {}
        for i, b in enumerate(res.blocks):
            by_sec.setdefault(b.section_path or "(无章节)", []).append((i, b.kind, b.text))
        doc = {}
        for sec in sorted(by_sec):
            items = by_sec[sec]
            doc[sec] = {
                "blocks": len(items),
                "previews": [
                    {"idx": idx, "kind": kind, "text": text[:110]}
                    for idx, kind, text in items[:3]
                ],
            }
        result[f.name] = doc
    OUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    total_sec = sum(len(v) for v in result.values())
    print(f"导出完成：{len(result)} 篇文档 / {total_sec} 个章节路径 → {OUT.name}")


if __name__ == "__main__":
    main()
