"""解析层入口：按扩展名分发。

对外只暴露 parse_document / parse_text 两个函数，
调用方不需要知道具体是哪种格式、用了哪个库。
"""

from pathlib import Path

from app.core.logging import logger
from app.services.parsing import docx as docx_parser
from app.services.parsing import markdown as markdown_parser
from app.services.parsing import pdf as pdf_parser
from app.services.parsing import plain as plain_parser
from app.services.parsing.base import Block, ParseResult, read_text

_SUFFIX_MAP = {
    ".pdf": pdf_parser.parse,
    ".docx": docx_parser.parse,
    ".md": markdown_parser.parse,
    ".markdown": markdown_parser.parse,
    ".txt": plain_parser.parse,
}

# 这几种格式的解析器收的是文本，不是路径/流，需要先归一化
_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}

SUPPORTED_SUFFIXES = tuple(sorted(_SUFFIX_MAP))

# python-docx 读不了 1997-2003 的老 .doc，直接给出可操作的提示
_HINTS = {
    ".doc": "老版 .doc 格式不支持，请在 Word 里另存为 .docx 再上传",
    ".wps": "暂不支持 .wps，请另存为 .docx 或 .pdf",
}


def parse_document(filename: str, source) -> ParseResult:
    """解析一个文件。source 可以是文件路径，也可以是二进制流（如 UploadFile.file）。"""
    suffix = Path(filename).suffix.lower()

    if suffix in _HINTS:
        raise ValueError(_HINTS[suffix])
    if suffix not in _SUFFIX_MAP:
        raise ValueError(
            f"不支持的文件类型 {suffix or '(无扩展名)'}，目前支持：{'、'.join(SUPPORTED_SUFFIXES)}"
        )

    if suffix in _TEXT_SUFFIXES:
        result = _SUFFIX_MAP[suffix](read_text(source))
    else:
        result = _SUFFIX_MAP[suffix](source)
    logger.info(
        "解析完成 file=%s 类型=%s 文本块=%d 页数=%s 需OCR=%s",
        filename,
        result.source_type,
        len(result.blocks),
        result.page_count,
        result.needs_ocr,
    )
    return result


def parse_text(content: str) -> ParseResult:
    """解析直接传进来的纯文本（JSON 上传走这条路）。"""
    return plain_parser.parse(content)


__all__ = ["Block", "ParseResult", "parse_document", "parse_text", "SUPPORTED_SUFFIXES"]
