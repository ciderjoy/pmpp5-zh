#!/usr/bin/env python3
"""把译稿中某个 figure 环境整体替换为原书裁剪图。

用法:
    .venv/bin/python tools/replace_figures.py chapters/07-convolution.tex 7-1 7-2 ...
    .venv/bin/python tools/replace_figures.py chapters/07-convolution.tex --all-from-crops

行为:
- 按 \\label{fig:<id>} 定位所在的 figure 环境；
- 读取并清洗 caption，去掉“依据原书图 X.Y 重绘”一类说明；
- 依据裁剪图长宽比选择合适的 \\includegraphics 宽度；
- 只替换目标 figure，其余内容保持不变。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURE_ENV_RE = re.compile(r"\\begin\{figure\}.*?\\end\{figure\}", re.S)
TABLE_ENV_RE = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S)

# “依据原书图 7.1 重绘”“依据原书图 18.6 逐字符重排”“依据原书练习 1 所附未编号有向图重绘”等说明
REDRAW_RE = re.compile(
    r"依据原书\s*(?:图|表|练习)\s*[A-Z]?\.?\d+(?:\.\d+)?\s*"
    r"(?:所附未编号有向图)?\s*(?:临时)?(?:逐字符)?(?:重绘|重排)[；;。]?"
)


def read_balanced(text: str, start: int) -> tuple[str, int]:
    """从 text[start] 的 '{' 开始读取平衡花括号内容，返回 (内容, 结束位置)。"""
    assert text[start] == "{"
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError("花括号不平衡")


def extract_caption(block: str) -> str | None:
    match = re.search(r"\\caption", block)
    if not match:
        return None
    index = match.end()
    while index < len(block) and block[index] in " \t\n":
        index += 1
    if index >= len(block) or block[index] != "{":
        return None
    caption, _ = read_balanced(block, index)
    return caption


def clean_caption(caption: str) -> str:
    text = REDRAW_RE.sub("", caption)
    # 去掉“，参数说明译为中文。”这类随重绘说明一起出现的补充语
    text = re.sub(r"[，,]\s*(?:参数)?说明(?:文字)?译为中文[；;。]?", "", text)
    # 清理空括号、多余空白与悬挂标点
    text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = text.strip()
    text = re.sub(r"[，、；;]\s*$", "", text)
    return text.strip()


def image_width_factor(image_path: Path) -> str:
    with fitz.open(image_path) as document:
        page = document[0]
        width, height = page.rect.width, page.rect.height
    if width <= 0:
        return r"\textwidth"
    aspect = height / width
    if aspect >= 1.5:
        factor = 0.55
    elif aspect >= 1.1:
        factor = 0.65
    elif aspect >= 0.85:
        factor = 0.75
    elif aspect >= 0.6:
        factor = 0.85
    else:
        factor = 1.0
    return r"\textwidth" if factor >= 1.0 else f"{factor:g}\\textwidth"


def build_replacement(label_name: str, image_rel: str, caption: str, width: str) -> str:
    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        f"  \\includegraphics[width={width}]{{{image_rel}}}",
    ]
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label_name}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def replace_in_file(
    chapter_path: Path, figure_ids: list[str], crops: dict
) -> list[str]:
    text = chapter_path.read_text(encoding="utf-8")
    replaced = []
    for figure_id in figure_ids:
        entry = crops.get(figure_id)
        if entry is None:
            print(f"  [跳过] {figure_id}: 裁剪清单中没有该条目", file=sys.stderr)
            continue
        image_path = PROJECT_ROOT / entry["output"]
        if not image_path.exists():
            print(f"  [跳过] {figure_id}: 图片不存在 {image_path}", file=sys.stderr)
            continue

        label_pattern = re.compile(
            re.escape(f"\\label{{fig:{figure_id}}}"), re.IGNORECASE
        )
        target = None
        for match in FIGURE_ENV_RE.finditer(text):
            if label_pattern.search(match.group(0)):
                target = match
                break
        if target is None:
            print(f"  [跳过] {figure_id}: 找不到对应 figure 环境", file=sys.stderr)
            continue

        block = target.group(0)
        caption = extract_caption(block)
        cleaned = clean_caption(caption) if caption else ""
        label_match = re.search(r"\\label\{(fig:[^}]*)\}", block)
        label_name = label_match.group(1) if label_match else f"fig:{figure_id}"
        width = image_width_factor(image_path)
        replacement = build_replacement(label_name, entry["output"], cleaned, width)
        text = text[: target.start()] + replacement + text[target.end() :]
        replaced.append(figure_id)
    chapter_path.write_text(text, encoding="utf-8")
    return replaced


def clean_all_captions(chapter_path: Path) -> int:
    """清洗文件内所有 figure/table 图注中的“重绘/重排”说明，保留代码图与表格正文。"""
    text = chapter_path.read_text(encoding="utf-8")
    count = 0
    matches = list(FIGURE_ENV_RE.finditer(text)) + list(TABLE_ENV_RE.finditer(text))
    for match in sorted(matches, key=lambda m: m.start(), reverse=True):
        block = match.group(0)
        caption = extract_caption(block)
        if caption is None or not REDRAW_RE.search(caption):
            continue
        cleaned = clean_caption(caption)
        start = block.index("\\caption")
        brace = block.index("{", start)
        _, end = read_balanced(block, brace)
        new_block = block[:start] + f"\\caption{{{cleaned}}}" + block[end:]
        text = text[: match.start()] + new_block + text[match.end() :]
        count += 1
    chapter_path.write_text(text, encoding="utf-8")
    return count


def load_crops() -> dict:
    manifest = json.loads((PROJECT_ROOT / "tools" / "figure-crops.json").read_text(encoding="utf-8"))
    return {figure["id"]: figure for figure in manifest["figures"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=Path)
    parser.add_argument("figure_ids", nargs="*")
    parser.add_argument(
        "--all-from-crops",
        action="store_true",
        help="替换该章在裁剪清单中出现的全部图号",
    )
    parser.add_argument(
        "--clean-captions-only",
        action="store_true",
        help="只清洗图注中的“重绘/重排”说明，不替换 figure",
    )
    args = parser.parse_args()

    chapter_path = args.chapter
    if not chapter_path.exists():
        raise SystemExit(f"找不到章节文件: {chapter_path}")

    if args.clean_captions_only:
        count = clean_all_captions(chapter_path)
        print(f"{chapter_path.name}: 清洗 {count} 条图注")
        return 0

    crops = load_crops()
    if args.all_from_crops:
        chapter_key = chapter_path.stem.split("-")[0]
        figure_ids = [
            figure_id
            for figure_id in crops
            if figure_id.startswith(f"{chapter_key}-")
            or figure_id.startswith(f"{chapter_key}-sidebar")
        ]
    else:
        figure_ids = args.figure_ids

    if not figure_ids:
        raise SystemExit("没有指定要替换的图号")

    replaced = replace_in_file(chapter_path, figure_ids, crops)
    print(f"{chapter_path.name}: 替换 {len(replaced)} 个 figure -> {', '.join(replaced)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
