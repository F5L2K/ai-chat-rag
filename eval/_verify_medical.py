# -*- coding: utf-8 -*-
"""验证医学语料：用项目自己的解析器复跑 4 个 md，确认章节路径全覆盖。

与 build_medical_corpus.py 的生成路径不同（解析器读 md → 切分章节），
交叉验证标题层级是否正确落到了 section_path。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 项目根
sys.path.insert(0, str(ROOT))

from app.services.parsing import parse_document  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent / "corpus_medical"


def main() -> None:
    for f in sorted(CORPUS_DIR.glob("*.md")):
        res = parse_document(f.name, str(f))
        blocks = res.blocks
        no_sec = [b for b in blocks if not b.section_path]
        heading_blocks = [b for b in blocks if b.kind == "heading"]
        paths = sorted({b.section_path for b in blocks if b.section_path})
        print("=" * 60)
        print(f.name)
        print(f"  总块数={len(blocks)}  needs_ocr={res.needs_ocr}")
        print(f"  标题块={len(heading_blocks)}  无章节块={len(no_sec)}  章节路径数={len(paths)}")
        print("  章节路径样本:")
        for p in paths[:10]:
            print(f"    - {p}")
        if len(paths) > 10:
            print(f"    ... 共 {len(paths)} 个")
        print("  首 5 个文本块:")
        for b in blocks[:5]:
            print(f"    [{b.kind}] path={b.section_path!r}: {b.text[:45]}")


if __name__ == "__main__":
    main()
