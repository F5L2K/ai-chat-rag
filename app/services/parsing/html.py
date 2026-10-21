"""HTML 解析：只取可见正文，h1–h6 直接给出章节层级。

为什么用标准库 html.parser 而不是 BeautifulSoup：
  这里只需要「可见文本 + 标题层级 + 表格」，不需要 CSS 选择器或 DOM 遍历，
  标准库足够，且不新增依赖。HTMLParser 对不规范的网页也能容错解析。

处理的几个关键点：
  1. script / style / head 等非正文内容整体跳过（否则会把 JS 当正文入库）
  2. h1–h6 映射为层级 1–6，喂给 SectionTracker 产出章节路径
  3. 块级标签作为段落边界，避免整页文字挤成一块
  4. table 转 Markdown 表格，保留行列关系（与 PDF 表格口径一致）
  5. <meta charset> 识别编码，避免 GBK 页面按 UTF-8 读出乱码
"""

import re
from html.parser import HTMLParser
from pathlib import Path

from app.services.parsing.base import (
    Block,
    ParseResult,
    SectionTracker,
    table_to_markdown,
)

# 这些标签的内容不是正文，整段跳过（head 里还有 title/meta/link，也一并跳过）
_SKIP_TAGS = {"head", "script", "style", "noscript", "template", "svg", "math"}

_HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}

# 块级标签：开始/结束时收尾当前段落，保证每段单独成块
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "header", "hr", "li",
    "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot",
    "th", "thead", "tr", "ul",
}

_META_CHARSET = re.compile(
    rb"<meta[^>]+charset\s*=\s*[\"']?\s*([a-zA-Z0-9_\-]+)", re.IGNORECASE
)


def _read_html(source) -> str:
    """把路径 / 二进制流 / 已是文本统一读成字符串，并按页面声明的编码解码。"""
    if isinstance(source, Path):
        data = source.read_bytes()
    elif isinstance(source, str):
        if "\n" in source or "\r" in source or "<" in source:
            return source
        p = Path(source)
        if p.exists() and p.is_file():
            data = p.read_bytes()
        else:
            return source
    else:
        data = source.read()
        if isinstance(data, str):
            return data

    # 先按 <meta charset> 声明解码，再依次兜底 utf-8 / gb18030
    encodings: list[str] = []
    m = _META_CHARSET.search(data[:4096])
    if m:
        declared = m.group(1).decode("ascii", "ignore").lower()
        # gb2312 / gbk 都是 gb18030 的子集，统一用它解码更稳
        encodings.append(
            "gb18030" if declared in ("gb2312", "gbk", "gb18030") else declared
        )
    encodings += ["utf-8", "gb18030"]

    for enc in encodings:
        try:
            return data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="ignore")


class _Extractor(HTMLParser):
    """把 HTML 事件流累积成 Block 列表。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.tracker = SectionTracker()
        self._skip_depth = 0
        self._buf: list[str] = []
        self._in_pre = False
        self._pre_buf: list[str] = []
        # 表格状态：只在最外层表格（depth==1）收集行列，嵌套表格的文本并入单元格
        self._table_depth = 0
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    # ---------- 文本累积 ----------

    def _take_text(self) -> str:
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        return text

    def _flush(self, prefix: str = "", kind: str = "text"):
        text = self._take_text()
        if text:
            self.blocks.append(
                Block(prefix + text, None, self.tracker.path, kind)
            )

    # ---------- HTMLParser 回调 ----------

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        # 容错：页面漏写 </head> 时，<body> 出现即结束跳过状态，避免整篇被吞掉
        if tag == "body" and self._skip_depth:
            self._skip_depth = 0
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag == "pre":
            self._flush()
            self._in_pre = True
            return

        if tag == "table":
            if self._table_depth == 0:
                self._flush()
                self._rows = []
            self._table_depth += 1
            return

        if self._table_depth:
            if self._table_depth == 1:
                if tag == "tr":
                    self._row = []
                elif tag in ("td", "th"):
                    self._cell = []
            return

        if tag in _HEADING_TAGS:
            self._flush()
            return

        if tag == "li":
            self._flush()
            return

        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return

        if tag == "pre":
            text = "".join(self._pre_buf).strip()
            if text:
                self.blocks.append(
                    Block(text, None, self.tracker.path, "text")
                )
            self._pre_buf = []
            self._in_pre = False
            return

        if tag == "table":
            if self._table_depth:
                self._table_depth -= 1
            if self._table_depth == 0:
                md = table_to_markdown(self._rows or [])
                if md:
                    self.blocks.append(
                        Block(md, None, self.tracker.path, "table")
                    )
                self._rows = self._row = self._cell = None
            return

        if self._table_depth:
            if self._table_depth == 1:
                if tag in ("td", "th") and self._cell is not None:
                    if self._row is not None:
                        self._row.append(" ".join(self._cell))
                    self._cell = None
                elif tag == "tr" and self._row is not None:
                    if any(cell.strip() for cell in self._row):
                        self._rows.append(self._row)  # type: ignore[union-attr]
                    self._row = None
            return

        if tag in _HEADING_TAGS:
            title = self._take_text()
            if title:
                level = _HEADING_TAGS[tag]
                self.tracker.push(level, title)
                self.blocks.append(
                    Block(title, None, self.tracker.path, "heading")
                )
            return

        if tag == "li":
            text = self._take_text()
            if text:
                # 列表项加符号，模型能看出这是并列条目而不是连续段落
                self.blocks.append(
                    Block("- " + text, None, self.tracker.path, "text")
                )
            return

        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_pre:
            self._pre_buf.append(data)
            return
        if self._table_depth:
            if self._table_depth == 1 and self._cell is not None:
                self._cell.append(data)
            return
        self._buf.append(data)

    def close(self):
        super().close()
        # 兜底：文档缺少闭合标签时，把残留文本收尾
        if self._table_depth and self._rows:
            md = table_to_markdown(self._rows)
            if md:
                self.blocks.append(Block(md, None, self.tracker.path, "table"))
            self._table_depth = 0
        self._flush()


def parse(source) -> ParseResult:
    extractor = _Extractor()
    extractor.feed(_read_html(source))
    extractor.close()
    return ParseResult(blocks=extractor.blocks, source_type="html", page_count=None)
