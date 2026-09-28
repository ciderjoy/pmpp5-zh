#!/usr/bin/env python3
"""把某一章的候选图拼成带编号的总览图，便于一次核对整章裁剪质量。

用法:
    .venv/bin/python tools/figure_montage.py ch7 [--cols 3] [--cell 420x330]
输出:
    build/figure-candidates/<chapter>/montage.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_ROOT = PROJECT_ROOT / "build" / "figure-candidates"


def parse_cell(text: str) -> tuple[float, float]:
    try:
        width, height = text.lower().split("x")
        return float(width), float(height)
    except ValueError as exc:  # pragma: no cover - CLI 参数错误
        raise argparse.ArgumentTypeError("cell 形如 420x330") from exc


def build_montage(chapter: str, cols: int, cell_w: float, cell_h: float) -> Path:
    chapter_dir = CANDIDATE_ROOT / chapter
    manifest = chapter_dir / "candidates.json"
    if not manifest.exists():
        raise SystemExit(f"找不到候选清单: {manifest}")

    figures = json.loads(manifest.read_text(encoding="utf-8"))["figures"]
    entries = []
    for figure in figures:
        image_path = chapter_dir / f"{figure['id']}.png"
        if image_path.exists():
            entries.append((figure["id"], image_path))
    if not entries:
        raise SystemExit(f"{chapter} 没有可用的候选图")

    rows = (len(entries) + cols - 1) // cols
    margin = 18.0
    label_h = 22.0
    page_w = margin * 2 + cols * cell_w
    page_h = margin * 2 + rows * (cell_h + label_h)

    document = fitz.open()
    page = document.new_page(width=page_w, height=page_h)
    page.draw_rect(fitz.Rect(0, 0, page_w, page_h), color=None, fill=(1, 1, 1))

    for index, (figure_id, image_path) in enumerate(entries):
        row, col = divmod(index, cols)
        x0 = margin + col * cell_w
        y0 = margin + row * (cell_h + label_h)
        label_rect = fitz.Rect(x0, y0, x0 + cell_w, y0 + label_h)
        page.insert_textbox(
            label_rect,
            f"图 {figure_id}",
            fontsize=15,
            fontname="helv",
            color=(0, 0, 0.8),
        )
        image_rect = fitz.Rect(x0, y0 + label_h, x0 + cell_w, y0 + label_h + cell_h)
        page.draw_rect(image_rect, color=(0.75, 0.75, 0.75), width=0.6)
        # 等比缩放并居中
        with fitz.open(image_path) as source:
            src_page = source[0]
            src_w, src_h = src_page.rect.width, src_page.rect.height
        if src_w <= 0 or src_h <= 0:
            continue
        scale = min(image_rect.width / src_w, image_rect.height / src_h)
        draw_w, draw_h = src_w * scale, src_h * scale
        target = fitz.Rect(
            image_rect.x0 + (image_rect.width - draw_w) / 2,
            image_rect.y0 + (image_rect.height - draw_h) / 2,
            image_rect.x0 + (image_rect.width + draw_w) / 2,
            image_rect.y0 + (image_rect.height + draw_h) / 2,
        )
        page.insert_image(target, filename=str(image_path))

    output = chapter_dir / "montage.png"
    page.get_pixmap(dpi=110).save(output)
    document.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", help="章节目录名，例如 ch7")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--cell", type=parse_cell, default=(430.0, 330.0))
    args = parser.parse_args()

    output = build_montage(args.chapter, args.cols, *args.cell)
    print(f"montage -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
