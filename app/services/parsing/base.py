"""解析层的公共数据结构与工具。

三种格式（PDF / Word / Markdown）的解析器都产出同样的 Block 列表，
这样切分层只需要面对一种数据形状。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


def read_text(source) -> str:
    """把「路径 / 二进制流 / 已是文本」统一读成字符串。

    Markdown 与纯文本解析器只认文本，但调用方可能给的是路径或上传流，
    这里做一次归一化，避免把路径字符串当成正文。
    """
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8", errors="ignore")
    if isinstance(source, str):
        # 含换行的一律当正文；否则才可能是路径
        if "\n" in source or "\r" in source:
            return source
        p = Path(source)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="ignore")
        return source
    data = source.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    return data


@dataclass
class Block:
    """解析出的最小文本单元，通常是一个段落、一个标题行或一张表格。"""

    text: str
    page_no: int | None = None
    section_path: str | None = None
    kind: str = "text"  # text / heading / table


@dataclass
class ParseResult:
    blocks: list[Block] = field(default_factory=list)
    source_type: str = "text"
    page_count: int | None = None
    needs_ocr: bool = False  # 疑似扫描件：整篇几乎提取不到文字


# ==========标题识别==========
# PDF 没有标题语义，只能靠「短行 + 编号模式」猜。
# Word 有 Heading 样式、Markdown 有 #，这两个走精确路径，不用这套启发式。
_CHAPTER = re.compile(r"^第\s*[一二三四五六七八九十百零\d]+\s*[章篇部]")
_SECTION = re.compile(r"^第\s*[一二三四五六七八九十百零\d]+\s*[节条]")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\s*[、.．]?\s+\S")
_CN_ORDER = re.compile(r"^[一二三四五六七八九十]{1,3}\s*[、.．]\s*\S")
_PAREN = re.compile(r"^[（(]\s*[一二三四五六七八九十\d]{1,3}\s*[)）]")

# 标题一般不长，也不会用句读结尾
_HEADING_MAX_LEN = 40
_SENTENCE_END = ("。", "，", "；", "：", "！", "？")


def heading_level(line: str) -> int | None:
    """推断一行标题的层级，不是标题则返回 None。层级只用于排序与嵌套，不要求绝对准确。"""
    s = line.strip()
    if not s or len(s) > _HEADING_MAX_LEN:
        return None
    if _CHAPTER.match(s):
        return 1
    if _SECTION.match(s):
        return 2
    m = _NUMBERED.match(s)
    if m:
        # 1.1 → 2 级，1.1.1 → 3 级
        return m.group(1).count(".") + 1
    if _CN_ORDER.match(s):
        return 2
    if _PAREN.match(s):
        return 3
    return None


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or s.endswith(_SENTENCE_END):
        return False
    return heading_level(s) is not None


class SectionTracker:
    """维护标题栈，产出「第3章 > 3.2 年假」这样的章节路径。

    push 时会弹掉层级不低于自己的旧标题，所以同级标题会互相替换，
    高级标题会保留，最终得到的就是从顶层到当前标题的一条链。
    """

    def __init__(self):
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str):
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title.strip()))

    @property
    def path(self) -> str | None:
        if not self._stack:
            return None
        return " > ".join(title for _, title in self._stack)


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """把二维表格转成 Markdown 表格，方便模型读懂行列关系。"""
    if not rows or not rows[0]:
        return ""
    cleaned = [
        [(cell or "").replace("\n", " ").strip() for cell in row] for row in rows
    ]
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    header = cleaned[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
