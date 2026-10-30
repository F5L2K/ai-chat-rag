# -*- coding: utf-8 -*-
"""临时脚本：导出传染病题库指定块范围全文，供标注集设计。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.parsing import parse_document  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus_medical"

RANGES = {
    "传染病题库.md": [(1, 36), (150, 166), (223, 242)],
}


def main() -> None:
    for fname, ranges in RANGES.items():
        res = parse_document(fname, str(CORPUS / fname))
        print("=" * 70)
        print(fname)
        for lo, hi in ranges:
            for i in range(lo, hi):
                if i < len(res.blocks):
                    b = res.blocks[i]
                    print(f"[{i}|{b.kind}|{b.section_path}] {b.text[:220]}")
            print("-" * 40)


if __name__ == "__main__":
    main()
