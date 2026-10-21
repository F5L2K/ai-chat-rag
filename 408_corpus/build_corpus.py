# -*- coding: utf-8 -*-
"""把开源 408 笔记仓库的 Markdown 转成 AIChat 知识库语料（按科目合并为一个 .md）。

来源：
  1. CSPostgraduate-408 的 408Notes/ 目录（KimYangOfCat/CSPostgraduate-408, MIT 类开源）
  2. cs-408（ddy-ddy）—— 经检查其笔记主体是手写扫描 PDF / OneNote 二进制，无文本，
     故不产出（见 语料说明）。

处理规则：
  - 剔除空占位文件（"一片荒芜" 或清洗后正文 < 100 字符）
  - 剔除 front-matter（--- title: ... ---）与纯图片行、纯 URL 行
  - 每科目按章节目录顺序合并为一个 UTF-8 无 BOM 的 .md
  - 保留原有 ATX 标题层级，保证 AIChat 的章节感知切块能吃到 section_path
"""

import re
import shutil
from pathlib import Path

CLONE_NOTES = Path(r"C:\Users\Administrator\PycharmProjects\AIChat\408_corpus\_tmp_clone\CSPostgraduate-408\408Notes")
OUT_DIR = Path(r"C:\Users\Administrator\PycharmProjects\AIChat\408_corpus\语料_408")

# 科目 -> (源目录, 合并文件名, 顶部标题)
SUBJECTS = [
    ("DataStructure",      "408_数据结构_笔记.md",      "数据结构"),
    ("操作系统",           "408_操作系统_笔记.md",      "操作系统"),
    ("操作系统错题集",     "408_操作系统_错题集.md",    "操作系统错题集"),
]

PLACEHOLDER = re.compile(r"一片荒芜")
IMAGE_LINE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
URL_LINE = re.compile(r"^\s*https?://\S+\s*$")
IMG_FRAG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_TAG = re.compile(r"<[^>]+>")
EMPTY_TABLE_ROW = re.compile(r"^\|[\s:|\-]+\|$")


def clean_md(text: str) -> str:
    """去 front-matter、图片（整行或作为标题/内联）、URL、HTML 标签，压缩空行。"""
    lines = text.splitlines()
    in_fm = False
    fm_seen = False
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not fm_seen and s == "---":
            in_fm = True
            fm_seen = True
            continue
        if in_fm:
            if s == "---":
                in_fm = False
            continue
        if IMAGE_LINE.match(ln) or URL_LINE.match(ln):
            continue
        line = IMG_FRAG.sub("", ln)
        line = HTML_TAG.sub("", line)
        line = line.rstrip()
        if not line.strip():
            continue
        if re.match(r"^\s*[-*+]\s*$", line):  # 图片被剥掉后剩下的空 bullet
            continue
        if EMPTY_TABLE_ROW.match(line.strip()):
            continue
        out.append(line)
    body = "\n".join(out)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def collect(subdir: str) -> list[tuple[str, str]]:
    """返回 [(相对路径, 清洗后文本)]，按路径排序，剔除占位与过短文件。"""
    items = []
    for p in sorted((CLONE_NOTES / subdir).rglob("*.md")):
        raw = p.read_text(encoding="utf-8", errors="replace")
        if PLACEHOLDER.search(raw):
            continue
        cleaned = clean_md(raw)
        if len(cleaned) < 100:  # 只有标题+截图链接的文件对文本检索无意义
            continue
        rel = p.relative_to(CLONE_NOTES)
        items.append((str(rel), cleaned))
    return items


def merge(subdir: str, title: str) -> tuple[str, str]:
    items = collect(subdir)
    parts = [f"# {title}\n"]
    for i, (rel, text) in enumerate(items):
        parts.append(f"<!-- 来源文件: {rel} -->\n{text}")
    return "\n\n".join(parts) + "\n", str(len(items))


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    stats = []
    for subdir, fname, title in SUBJECTS:
        merged, n = merge(subdir, title)
        (OUT_DIR / fname).write_text(merged, encoding="utf-8")
        stats.append((fname, n, len(merged)))

    # 复习总览：取 408Notes/README.md（科目介绍/教材/真题/笔记说明）
    overview_raw = (CLONE_NOTES / "README.md").read_text(encoding="utf-8", errors="replace")
    overview = clean_md(overview_raw)
    overview = "# 408 复习总览（来源仓库项目说明）\n\n" + overview
    (OUT_DIR / "408_复习总览.md").write_text(overview, encoding="utf-8")
    stats.append(("408_复习总览.md", 1, len(overview)))

    print("== 产出 ==")
    for fname, n, chars in stats:
        print(f"{fname}: {n} 个源文件, {chars} 字符")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
