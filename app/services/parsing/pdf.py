"""PDF 解析：pdfplumber 逐页取文本与表格，扫描页自动走 OCR 兜底。

选 pdfplumber 而不是 PyMuPDF，是因为 extract_tables() 是同类里最强的，
而「表格转 Markdown」是需求之一。代价是速度慢一些，当前文档量可以接受。

扫描件兜底：某页 extract_text 出来少于 _EMPTY_PAGE_CHARS 字时，
用 PyMuPDF 把该页渲染成图，丢给 OCR 模块识别。OCR 未装或失败时
自动降级为「该页无文字、整篇标 needs_ocr」，不影响主流程。
"""

import io
from collections.abc import Iterator

import pdfplumber

from app.core.logging import logger
from app.services.parsing import ocr as ocr_service
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
# 超过这个比例的页没有文字（且 OCR 也没救回来），整篇判定为扫描件
_OCR_PAGE_RATIO = 0.5


def _extract_page_blocks(
    text: str,
    page_no: int,
    tracker: SectionTracker,
) -> Iterator[Block]:
    """把一页的纯文本按行切成 Block，标题行推入 SectionTracker。"""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        level = heading_level(line) if looks_like_heading(line) else None
        if level is not None:
            tracker.push(level, line)
            yield_block = Block(line, page_no, tracker.path, "heading")
        else:
            yield_block = Block(line, page_no, tracker.path, "text")
        yield yield_block


def parse(source) -> ParseResult:
    blocks: list[Block] = []
    tracker = SectionTracker()
    empty_pages = 0
    ocr_recovered = 0
    page_count = 0

    # source 可能是路径 / BytesIO。pdfplumber 直接吃；OCR 渲染需要原始 bytes，
    # 这里先整包读出来，再包成 BytesIO 给 pdfplumber，避免重复读流。
    if hasattr(source, "read"):
        raw = source.read()
    else:
        with open(source, "rb") as f:
            raw = f.read()
    bio = io.BytesIO(raw)

    with pdfplumber.open(bio) as pdf:
        page_count = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            page_is_empty = len(text.strip()) < _EMPTY_PAGE_CHARS

            if page_is_empty:
                empty_pages += 1
                # 尝试 OCR 把这页救回来
                ocr_text = ""
                image = ocr_service.render_pdf_page(raw, page_no)
                if image is not None:
                    ocr_text = ocr_service.recognize(image).strip()
                if ocr_text:
                    ocr_recovered += 1
                    logger.info(
                        "第 %d 页文字层为空，OCR 识别出 %d 字",
                        page_no,
                        len(ocr_text),
                    )
                    text = ocr_text

            # PDF 的换行是排版换行，逐行处理，段落合并交给切分层
            for line_block in _extract_page_blocks(text, page_no, tracker):
                blocks.append(line_block)

            # 表格单独提取：extract_text 会把表格压成一行串，行列关系全丢
            # OCR 出来的页没有可抽的矢量表格，跳过
            if not page_is_empty:
                for table in page.extract_tables() or []:
                    md = table_to_markdown(table)
                    if md:
                        blocks.append(Block(md, page_no, tracker.path, "table"))

    # OCR 救不回来的空页仍占多数，才判定为扫描件
    unresolved_empty = empty_pages - ocr_recovered
    needs_ocr = (
        page_count > 0
        and unresolved_empty > 0
        and (unresolved_empty / page_count) >= _OCR_PAGE_RATIO
    )
    if ocr_recovered:
        logger.info(
            "PDF 扫描页 OCR 兜底成功 %d/%d 页", ocr_recovered, empty_pages
        )
    return ParseResult(
        blocks=blocks,
        source_type="pdf",
        page_count=page_count,
        needs_ocr=needs_ocr,
    )
