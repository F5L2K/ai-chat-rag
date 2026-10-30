# -*- coding: utf-8 -*-
"""医学题库语料预处理：识别标题结构 → 输出带 ATX 层级的 Markdown。

背景：3 个 docx 全部段落是 Word 的 Normal 样式（无 Heading），直接入库会让
解析器丢失章节路径（引用定位失效、SecHit 无法评估）。本脚本按文档固有结构
（「第X章 / 一、 / 名词解释：」）把标题段落转成 # 层级，输出可直接入库、
可直接用于检索评估的 Markdown 语料。

输出目录：eval/corpus_medical/（与 eval/corpus_kb 等并列）
验证方式：用 app.services.parsing.parse_document 复跑，确认全部块带 section_path
（见同目录 _verify_medical.py）。
"""

import re
import sys
from pathlib import Path

# 源文件（用户桌面）
SRC_DIR = Path(r"C:\Users\Administrator\Desktop")
# 输出目录
OUT_DIR = Path(__file__).resolve().parent / "corpus_medical"

# ========== 标题识别规则 ==========

# H1：章级（第一章 肺系病证）
_H1_CHAPTER = re.compile(r"^第[一二三四五六七八九十百零]+[章篇]")

# H2：中文序号题型块（一、填空题 / 二、单项选择题： / 六、论述题（10分））
_H2_CN = re.compile(r"^[一二三四五六七八九十]+、")

# H2：名词解释： / 填空题： / 简答题： / 论述题 等块级标题（整行只含词 + 可选冒号/分）
_H2_FULL = re.compile(
    r"^(名词解释|填空题|简答题|单项选择题|多选题|判断题|论述题|病案分析|"
    r"病例分析|案例分析题|问答题|配伍题|单项选择题)[:：]?"
    r"(?:\s*[（(]?\s*\d*\s*分?\s*[)）]?)?$"
)


def classify(line: str, is_first: bool) -> int:
    """返回标题层级：1=H1, 2=H2, 0=正文。"""
    s = line.strip()
    if not s:
        return 0
    if _H1_CHAPTER.match(s):
        return 1
    if _H2_CN.match(s) or _H2_FULL.match(s):
        return 2
    # 文档首行的短标题（如「推拿治疗学」）当作 H1；
    # 首行若是题型块标题（如「名词解释：」）已被上面规则接住，不会走到这里。
    if is_first and len(s) <= 20:
        return 1
    return 0


def _to_md_lines(lines: list[str]) -> list[str]:
    """把文本行按标题规则转成 md 行。"""
    out: list[str] = []
    first = True
    for raw in lines:
        t = raw.strip()
        if not t:
            continue
        lvl = classify(t, first)
        first = False
        if lvl == 1:
            out.append(f"# {t}")
        elif lvl == 2:
            out.append(f"## {t}")
        else:
            out.append(t)
    return out


def docx_to_md(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    lines = [p.text for p in d.paragraphs if p.text.strip()]
    return "\n\n".join(_to_md_lines(lines))


def pdf_to_md(path: Path) -> str:
    import pymupdf  # fitz 的新包名；旧接口 fitz 仍可用

    doc = pymupdf.open(str(path))
    lines: list[str] = []
    for page in doc:
        for line in page.get_text().splitlines():
            if line.strip():
                lines.append(line)
    return "\n\n".join(_to_md_lines(lines))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    jobs = [
        ("中医内科习题集（整理）.docx", "中医内科习题集.md", docx_to_md),
        ("推拿题库（选+填）.docx", "推拿题库.md", docx_to_md),
        ("西医内科（题库）.docx", "西医内科.md", docx_to_md),
        ("传染病题库1.pdf", "传染病题库.md", pdf_to_md),
    ]
    for src, dst, fn in jobs:
        src_path = SRC_DIR / src
        if not src_path.exists():
            print(f"[跳过] 找不到源文件: {src_path}")
            continue
        text = fn(src_path)
        dst_path = OUT_DIR / dst
        # UTF-8 无 BOM；解析器 read_text 用 utf-8
        dst_path.write_text(text, encoding="utf-8", newline="\n")
        n_h1 = len(re.findall(r"^# ", text, re.M))
        n_h2 = len(re.findall(r"^## ", text, re.M))
        print(f"[生成] {dst_path.name}  字符={len(text)}  H1={n_h1} H2={n_h2}")


if __name__ == "__main__":
    main()
