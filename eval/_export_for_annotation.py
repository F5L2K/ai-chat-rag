# -*- coding: utf-8 -*-
"""导出第二批标注需要的语料原文：按选定 idx 范围导出块文本，供选题参考。

输出 eval/_tmp/{doc}_selection.txt，格式：
  [{idx}|{section_path}] {text 单行}
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.parsing import parse_document  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus_medical"
OUT = Path(__file__).resolve().parent / "_tmp"
OUT.mkdir(exist_ok=True)

# 每篇文档要导出的 idx 范围（闭区间）
RANGES = {
    "中医内科习题集.md": [
        (57, 120), (88, 117), (228, 249), (335, 368), (369, 390),
        (425, 435), (496, 519), (558, 582), (583, 614), (726, 760),
        (761, 803), (838, 850), (881, 899), (900, 922),
    ],
    "传染病题库.md": [
        (1, 149), (150, 222), (223, 420),
    ],
    "推拿题库.md": [
        (31, 130), (244, 330), (393, 411), (412, 446), (447, 466), (467, 496),
    ],
    "西医内科.md": [
        (0, 35), (36, 95), (130, 230),
    ],
}


def main() -> None:
    for fname, ranges in RANGES.items():
        res = parse_document(fname, str(CORPUS / fname))
        blocks = res.blocks
        lines = []
        for lo, hi in ranges:
            for i in range(lo, hi + 1):
                if i >= len(blocks):
                    break
                b = blocks[i]
                text = b.text.replace("\n", " ").strip()
                if not text:
                    continue
                lines.append(f"[{i}|{b.section_path}] {text}")
        out = OUT / f"{Path(fname).stem}_selection.txt"
        out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        print(f"{fname}: 导出 {len(lines)} 行 -> {out.name}")


if __name__ == "__main__":
    main()
