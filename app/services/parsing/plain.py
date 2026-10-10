"""纯文本解析：没有标题语义，整篇按行收进来，章节留空。"""

from app.services.parsing.base import Block, ParseResult


def parse(text: str) -> ParseResult:
    blocks = [
        Block(line.strip(), None, None, "text")
        for line in text.splitlines()
        if line.strip()
    ]
    return ParseResult(blocks=blocks, source_type="text", page_count=None)
