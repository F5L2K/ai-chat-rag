"""Markdown 解析：标题天然带层级，正则读即可，零额外依赖。

要注意代码块里的 # 不是标题，所以跟踪 ``` 围栏状态。
"""

import re

from app.services.parsing.base import Block, ParseResult, SectionTracker

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SETEXT_UNDERLINE = re.compile(r"^\s*(=+|-{2,})\s*$")


def parse(text: str) -> ParseResult:
    tracker = SectionTracker()
    blocks: list[Block] = []
    lines = text.splitlines()
    in_code = False
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        if _FENCE.match(line):
            in_code = not in_code
            blocks.append(Block(line, None, tracker.path, "text"))
            i += 1
            continue

        if in_code:
            blocks.append(Block(line, None, tracker.path, "text"))
            i += 1
            continue

        # ATX 标题：# / ## / ###
        m = _ATX.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            tracker.push(level, title)
            blocks.append(
                Block(f"{'#' * level} {title}", None, tracker.path, "heading")
            )
            i += 1
            continue

        # Setext 标题：正文行 + 下一行 === 或 ---
        if (
            line.strip()
            and i + 1 < len(lines)
            and _SETEXT_UNDERLINE.match(lines[i + 1])
        ):
            level = 1 if lines[i + 1].strip().startswith("=") else 2
            tracker.push(level, line.strip())
            blocks.append(Block(line.strip(), None, tracker.path, "heading"))
            i += 2
            continue

        if line.strip():
            blocks.append(Block(line.strip(), None, tracker.path, "text"))
        i += 1

    return ParseResult(blocks=blocks, source_type="md", page_count=None)
