"""检查一份纯文本/文档被解析 + 切分后的章节结构。

用法：
    python test_samples/check_structure.py test_samples/智慧办公系统_使用手册.txt

它只做「解析 -> 切块」两步，不写库、不调模型，所以秒出结果，
用来和「上传到知识库后」的实际落库结果做比对。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.chunking import build_chunks
from app.services.parsing import parse_document

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python test_samples/check_structure.py <文件路径>")
        return 1

    path = Path(sys.argv[1]).resolve()
    parsed = parse_document(path.name, str(path))
    chunks = build_chunks(parsed.blocks)

    print(f"文件：{path.name}")
    print(f"解析类型：{parsed.source_type}   文本块：{len(parsed.blocks)}   切块：{len(chunks)}")
    print()

    print("=== 识别出的标题 ===")
    headings = [b for b in parsed.blocks if b.kind == "heading"]
    for b in headings:
        print(f"  {b.text!r:<24} -> {b.section_path!r}")
    print()

    print("=== 切块结果（每块带章节路径）===")
    for i, c in enumerate(chunks):
        head = c.text.replace("\n", " ")[:44]
        print(f"  [{i:>2}] {c.section_path!r}")
        print(f"       {head}…")
    print()

    missing = [i for i, c in enumerate(chunks) if not c.section_path]
    if missing:
        print(f"⚠ 有 {len(missing)} 个块没有 section_path：{missing}")
    else:
        print("✓ 所有块都带上了 section_path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
