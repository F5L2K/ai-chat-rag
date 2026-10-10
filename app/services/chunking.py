"""章节感知切分。

和普通按长度切分的区别：这里先认章节和页码边界，再在边界内按长度切，
所以每一块都能回答「我来自哪一章、哪一页」——这是带引用回答的前提。

三种边界会触发切块：
  1. 章节变了      —— 但只在缓冲里已有正文时才切，否则「第一章」这类标题会单独成块
  2. 累积超长      —— 超过 chunk_size
  3. 页码变了      —— 但只在已经攒够字时才切，否则会产生一堆碎块
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE
from app.services.parsing.base import Block

# 页码变化时是否立刻收尾，取决于当前攒了多少字。
# 攒够了就切（页码准确），没攒够就继续攒（避免碎块）。
_MIN_CHUNK_CHARS = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)


@dataclass
class Chunk:
    text: str
    page_no: int | None
    section_path: str | None
    char_start: int
    char_end: int


def _split_oversized(text: str) -> list[str]:
    """单个文本块自己就超长时（比如一张大表格）先按长度切开。"""
    if len(text) <= RAG_CHUNK_SIZE:
        return [text]
    return _splitter.split_text(text)


def build_chunks(blocks: list[Block], chunk_size: int = RAG_CHUNK_SIZE) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    buffer_has_body = False  # 缓冲里是否已有正文（非标题），用来判断是否值得单独成块
    cur_section: str | None = None
    cur_page: int | None = None
    cur_start = 0
    cursor = 0  # 在整篇文档中的字符偏移，用来填 char_start / char_end

    def flush():
        nonlocal buffer, buffer_len, buffer_has_body
        if buffer:
            text = "\n".join(buffer)
            chunks.append(
                Chunk(
                    text=text,
                    page_no=cur_page,
                    section_path=cur_section,
                    char_start=cur_start,
                    char_end=cur_start + len(text),
                )
            )
        buffer = []
        buffer_len = 0
        buffer_has_body = False

    for block in blocks:
        text = block.text.strip()
        if not text:
            cursor += len(block.text) + 1
            continue

        for piece in _split_oversized(text):
            if buffer and (
                # 章节变化只在已攒到正文时才切；否则「第一章」这类标题会单独成块
                (block.section_path != cur_section and buffer_has_body)
                or buffer_len + len(piece) > chunk_size
                or (block.page_no != cur_page and buffer_len >= _MIN_CHUNK_CHARS)
            ):
                flush()

            if not buffer:
                cur_section = block.section_path
                cur_page = block.page_no
                cur_start = cursor

            buffer.append(piece)
            buffer_len += len(piece)
            if block.kind == "heading":
                # 标题是它后续内容的章节标签：即使缓冲非空（前一个标题还没配正文），
                # 也要把标签更新到最新层级，否则块内容与 section_path 会对不上
                cur_section = block.section_path
            else:
                buffer_has_body = True

        cursor += len(block.text) + 1

    flush()
    return chunks
