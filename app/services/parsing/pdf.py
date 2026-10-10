"""PDF 解析：pdfplumber 逐页取文本与表格。

选 pdfplumber 而不是 PyMuPDF，是因为 extract_tables() 是同类里最强的，
而「表格转 Markdown」是需求之一。代价是速度慢一些，当前文档量可以接受。
"""

import pdfplumber

from app.services.parsing.base import (
    Block,
    ParseResult,
    SectionTracker,
    heading_level,
    looks_like_heading,
    table_to_markdown,
)

# 一页提取到的字符少于这个数，就认为这页没有文字层（大概率是扫描图）
_EMPTY_PAGE_CHARS = 20
# 超过这个比例的页没有文字，整篇判定为扫描件
_OCR_PAGE_RATIO = 0.5


def parse(source) -> ParseResult:
    blocks: list[Block] = []
    tracker = SectionTracker()
    empty_pages = 0
    page_count = 0

    with pdfplumber.open(source) as pdf:
        page_count = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < _EMPTY_PAGE_CHARS:
                empty_pages += 1

            # PDF 的换行是排版换行，逐行处理，段落合并交给切分层
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                level = heading_level(line) if looks_like_heading(line) else None
                if level is not None:
                    tracker.push(level, line)
                    blocks.append(Block(line, page_no, tracker.path, "heading"))
                else:
                    blocks.append(Block(line, page_no, tracker.path, "text"))

            # 表格单独提取：extract_text 会把表格压成一行串，行列关系全丢
            for table in page.extract_tables() or []:
                md = table_to_markdown(table)
                if md:
                    blocks.append(Block(md, page_no, tracker.path, "table"))

    needs_ocr = page_count > 0 and (empty_pages / page_count) >= _OCR_PAGE_RATIO
    return ParseResult(
        blocks=blocks,
        source_type="pdf",
        page_count=page_count,
        needs_ocr=needs_ocr,
    )
