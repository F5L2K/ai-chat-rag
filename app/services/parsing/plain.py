"""纯文本解析：没有格式标记，靠编号模式猜标题层级。

纯文本是最容易「丢结构」的一条上传路径：用户从别处复制过来的内容，
往往自带 `## 第一章` 或 `一、年假规定` 这类标题，却没有任何格式契约。
这里复用 base 里的启发式标题识别（本来是为 PDF 写的），让纯文本也能产出
section_path —— 否则带引用回答永远只能引用到「整篇」，定位不到章节。
"""

import re

from app.services.parsing.base import (
    Block,
    ParseResult,
    SectionTracker,
    heading_level,
    looks_like_heading,
)

# 纯文本里也常混着 Markdown 式标题（复制粘贴的产物），一并识别
_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def parse(text: str) -> ParseResult:
    tracker = SectionTracker()
    blocks: list[Block] = []

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        # Markdown 式标题：层级以 # 为准。这是显式标记，优先于启发式猜测
        m = _ATX.match(s)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if title:
                tracker.push(level, title)
                blocks.append(
                    Block(f"{'#' * level} {title}", None, tracker.path, "heading")
                )
                continue

        # 编号式标题：第一章 / 一、 / （一） / 1.1
        level = heading_level(s)
        if level is not None and looks_like_heading(s):
            tracker.push(level, s)
            blocks.append(Block(s, None, tracker.path, "heading"))
            continue

        blocks.append(Block(s, None, tracker.path, "text"))

    return ParseResult(blocks=blocks, source_type="text", page_count=None)
