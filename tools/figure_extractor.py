#!/usr/bin/env python3
"""Batch-extract figures from the PMPP source PDF.

Two workflows are supported:

1. ``extract`` renders reviewed crop rectangles from figure-crops.json.
2. ``detect`` scans ``FIGURE x.y`` captions and proposes crop rectangles for
   human review. Automatic rectangles are candidates, not final truth.

All page numbers are one-based PDF physical page numbers. Crop coordinates in
the manifest are pixels at the manifest's reference DPI.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "figure-crops.json"
FIGURE_RE = re.compile(r"\bFIGURE\s+(\d+)\.(\d+)\b", re.IGNORECASE)


class FigureExtractorError(RuntimeError):
    """A user-facing extraction error."""


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FigureExtractorError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FigureExtractorError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FigureExtractorError(f"Expected a JSON object in {path}")
    return data


def project_root(manifest_path: Path, manifest: Dict[str, Any]) -> Path:
    configured = Path(str(manifest.get("project_root", "..")))
    if configured.is_absolute():
        return configured.resolve()
    return (manifest_path.resolve().parent / configured).resolve()


def source_pdf(
    manifest_path: Path, manifest: Dict[str, Any], override: Optional[Path]
) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    root = project_root(manifest_path, manifest)
    configured = Path(str(manifest.get("pdf", "")))
    if not configured:
        raise FigureExtractorError("Manifest is missing the 'pdf' field")
    return configured.resolve() if configured.is_absolute() else root / configured


def selected_figures(
    manifest: Dict[str, Any], requested_ids: Sequence[str]
) -> List[Dict[str, Any]]:
    figures = manifest.get("figures")
    if not isinstance(figures, list):
        raise FigureExtractorError("Manifest field 'figures' must be an array")
    by_id = {
        str(item.get("id")): item
        for item in figures
        if isinstance(item, dict) and item.get("id")
    }
    if not requested_ids:
        return list(by_id.values())
    missing = [figure_id for figure_id in requested_ids if figure_id not in by_id]
    if missing:
        raise FigureExtractorError(
            "Unknown figure ID(s): " + ", ".join(sorted(missing))
        )
    return [by_id[figure_id] for figure_id in requested_ids]


def crop_rect_points(crop: Dict[str, Any], dpi: int) -> fitz.Rect:
    required = ("x", "y", "width", "height")
    missing = [key for key in required if key not in crop]
    if missing:
        raise FigureExtractorError(
            "Crop is missing field(s): " + ", ".join(missing)
        )
    x = float(crop["x"])
    y = float(crop["y"])
    width = float(crop["width"])
    height = float(crop["height"])
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise FigureExtractorError(f"Invalid crop rectangle: {crop}")
    points_per_pixel = 72.0 / dpi
    return fitz.Rect(
        x * points_per_pixel,
        y * points_per_pixel,
        (x + width) * points_per_pixel,
        (y + height) * points_per_pixel,
    )


def crop_pixels(rect: fitz.Rect, dpi: int) -> Dict[str, int]:
    pixels_per_point = dpi / 72.0
    x0 = max(0, round(rect.x0 * pixels_per_point))
    y0 = max(0, round(rect.y0 * pixels_per_point))
    x1 = max(x0 + 1, round(rect.x1 * pixels_per_point))
    y1 = max(y0 + 1, round(rect.y1 * pixels_per_point))
    return {
        "x": x0,
        "y": y0,
        "width": x1 - x0,
        "height": y1 - y0,
    }


def resolved_output(
    root: Path, figure: Dict[str, Any], output_root: Optional[Path]
) -> Path:
    configured = Path(str(figure.get("output", "")))
    if not configured:
        raise FigureExtractorError(f"Figure {figure.get('id')} has no output path")
    if configured.is_absolute():
        return configured
    base = output_root.resolve() if output_root else root
    return base / configured


def render_clip(
    document: fitz.Document,
    page_number: int,
    rect: fitz.Rect,
    dpi: int,
    output: Path,
) -> Tuple[int, int]:
    if page_number < 1 or page_number > document.page_count:
        raise FigureExtractorError(
            f"PDF page {page_number} is outside 1..{document.page_count}"
        )
    page = document[page_number - 1]
    if rect.is_empty or not page.rect.contains(rect):
        raise FigureExtractorError(
            f"Crop {rect} is outside PDF page {page_number} bounds {page.rect}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=rect,
        alpha=False,
        annots=False,
    )
    pixmap.save(str(output))
    return pixmap.width, pixmap.height


def validate_manifest(
    document: fitz.Document,
    figures: Iterable[Dict[str, Any]],
    dpi: int,
) -> None:
    seen = set()
    for figure in figures:
        figure_id = str(figure.get("id", ""))
        if not figure_id:
            raise FigureExtractorError("A figure is missing its ID")
        if figure_id in seen:
            raise FigureExtractorError(f"Duplicate figure ID: {figure_id}")
        seen.add(figure_id)
        page_number = int(figure.get("page", 0))
        if page_number < 1 or page_number > document.page_count:
            raise FigureExtractorError(
                f"Figure {figure_id}: invalid PDF page {page_number}"
            )
        crop = figure.get("crop")
        if not isinstance(crop, dict):
            raise FigureExtractorError(f"Figure {figure_id}: crop must be an object")
        rect = crop_rect_points(crop, dpi)
        page_rect = document[page_number - 1].rect
        if rect.is_empty or not page_rect.contains(rect):
            raise FigureExtractorError(
                f"Figure {figure_id}: crop {crop} exceeds page bounds"
            )


def command_extract(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    root = project_root(manifest_path, manifest)
    pdf_path = source_pdf(manifest_path, manifest, args.pdf)
    if not pdf_path.is_file():
        raise FigureExtractorError(f"Source PDF not found: {pdf_path}")
    dpi = int(args.dpi or manifest.get("dpi", 300))
    figures = selected_figures(manifest, args.ids)
    output_root = args.output_root.expanduser().resolve() if args.output_root else None

    with fitz.open(pdf_path) as document:
        validate_manifest(document, figures, dpi)
        rendered: List[Tuple[str, Path]] = []
        for figure in figures:
            figure_id = str(figure["id"])
            output = resolved_output(root, figure, output_root)
            if output.exists() and not args.force:
                print(f"skip  {figure_id:>6}  {output} (use --force to overwrite)")
                rendered.append((figure_id, output))
                continue
            rect = crop_rect_points(figure["crop"], dpi)
            width, height = render_clip(
                document, int(figure["page"]), rect, dpi, output
            )
            print(f"write {figure_id:>6}  {width}x{height}  {output}")
            rendered.append((figure_id, output))

    if args.gallery:
        write_gallery(args.gallery.expanduser().resolve(), rendered, "Reviewed crops")
    return 0


def rects_touch(a: fitz.Rect, b: fitz.Rect, vertical_gap: float, x_gap: float) -> bool:
    horizontal_distance = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
    vertical_distance = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
    return horizontal_distance <= x_gap and vertical_distance <= vertical_gap


def page_elements_above_caption(
    page: fitz.Page, caption: fitz.Rect, minimum_top: float
) -> List[fitz.Rect]:
    elements: List[fitz.Rect] = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        bbox = block.get("bbox")
        if not bbox:
            continue
        rect = fitz.Rect(bbox)
        if rect.y0 >= minimum_top and rect.y1 <= caption.y0 + 1:
            elements.append(rect)
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if rect.y0 >= minimum_top and rect.y1 <= caption.y0 + 1:
            elements.append(rect)
    for image in page.get_image_info():
        rect = fitz.Rect(image["bbox"])
        if rect.y0 >= minimum_top and rect.y1 <= caption.y0 + 1:
            elements.append(rect)
    return [rect for rect in elements if rect.width > 0.5 and rect.height > 0.5]


def propose_figure_rect(
    page: fitz.Page,
    caption: fitz.Rect,
    padding: float,
    vertical_gap: float,
    minimum_top: float,
) -> fitz.Rect:
    elements = page_elements_above_caption(page, caption, minimum_top)
    if not elements:
        return fitz.Rect(
            page.rect.x0 + 36,
            max(minimum_top, caption.y0 - page.rect.height * 0.35),
            page.rect.x1 - 36,
            caption.y0 - 4,
        )

    seed_limit = caption.y0 - vertical_gap * 2.0
    selected = [rect for rect in elements if rect.y1 >= seed_limit]
    if not selected:
        nearest = max(elements, key=lambda rect: rect.y1)
        selected = [nearest]

    union = fitz.Rect(selected[0])
    for rect in selected[1:]:
        union |= rect

    changed = True
    while changed:
        changed = False
        for rect in elements:
            if rect in selected:
                continue
            if rects_touch(union, rect, vertical_gap, x_gap=18):
                selected.append(rect)
                union |= rect
                changed = True

    union = fitz.Rect(
        union.x0 - padding,
        union.y0 - padding,
        union.x1 + padding,
        min(caption.y0 - 2, union.y1 + padding),
    )
    return union & page.rect


def find_caption_rect(page: fitz.Page, figure_id: str) -> Optional[fitz.Rect]:
    queries = (f"FIGURE {figure_id}", f"Figure {figure_id}", f"figure {figure_id}")
    for query in queries:
        matches = page.search_for(query)
        if matches:
            return fitz.Rect(matches[0])
    return None


def detect_caption_ids(page: fitz.Page, chapter: Optional[int]) -> List[str]:
    found: List[str] = []
    for match in FIGURE_RE.finditer(page.get_text("text")):
        chapter_number = int(match.group(1))
        if chapter is not None and chapter_number != chapter:
            continue
        figure_id = f"{chapter_number}-{int(match.group(2))}"
        if figure_id not in found:
            found.append(figure_id)
    return found


def write_gallery(
    gallery_path: Path,
    images: Sequence[Tuple[str, Path]],
    title: str,
) -> None:
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for figure_id, image_path in images:
        relative = os.path.relpath(image_path, gallery_path.parent)
        cards.append(
            "<article>"
            f"<h2>{html.escape(figure_id)}</h2>"
            f'<img src="{html.escape(relative)}" loading="lazy">'
            f"<p>{html.escape(str(image_path))}</p>"
            "</article>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; background: #eee; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(360px,1fr)); gap: 1rem; }}
article {{ background: white; padding: 1rem; border: 1px solid #bbb; }}
img {{ display: block; max-width: 100%; max-height: 70vh; margin: auto; }}
p {{ overflow-wrap: anywhere; color: #555; font-size: .85rem; }}
</style>
</head>
<body><h1>{html.escape(title)}</h1><main>{''.join(cards)}</main></body>
</html>
"""
    gallery_path.write_text(document, encoding="utf-8")
    print(f"gallery       {gallery_path}")


def command_detect(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    root = project_root(manifest_path, manifest)
    pdf_path = source_pdf(manifest_path, manifest, args.pdf)
    if not pdf_path.is_file():
        raise FigureExtractorError(f"Source PDF not found: {pdf_path}")
    dpi = int(args.dpi or manifest.get("dpi", 300))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: List[Dict[str, Any]] = []
    previews: List[Tuple[str, Path]] = []

    with fitz.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page = document[page_index]
            captions: List[Tuple[str, fitz.Rect]] = []
            for figure_id in detect_caption_ids(page, args.chapter):
                caption = find_caption_rect(page, figure_id.replace("-", "."))
                if caption is None:
                    print(
                        f"warn  {figure_id:>6}  caption text found but rectangle unavailable",
                        file=sys.stderr,
                    )
                    continue
                captions.append((figure_id, caption))

            captions.sort(key=lambda item: (item[1].y0, item[1].x0))
            previous_caption: Optional[fitz.Rect] = None
            for figure_id, caption in captions:
                minimum_top = float(args.minimum_top)
                if previous_caption is not None:
                    boundary = previous_caption.y1 + float(args.caption_clearance)
                    if boundary < caption.y0:
                        minimum_top = max(minimum_top, boundary)
                rect = propose_figure_rect(
                    page,
                    caption,
                    padding=float(args.padding),
                    vertical_gap=float(args.vertical_gap),
                    minimum_top=minimum_top,
                )
                crop = crop_pixels(rect, dpi)
                preview = output_dir / f"{figure_id}.png"
                width, height = render_clip(
                    document, page_index + 1, rect, dpi, preview
                )
                print(
                    f"guess {figure_id:>6}  page {page_index + 1:>3}  "
                    f"{width}x{height}  {preview}"
                )
                chapter_number = figure_id.split("-", 1)[0]
                candidates.append(
                    {
                        "id": figure_id,
                        "page": page_index + 1,
                        "crop": crop,
                        "output": f"figs/ch{chapter_number}/{figure_id}.png",
                        "review": "candidate",
                    }
                )
                previews.append((figure_id, preview))
                previous_caption = caption

    candidate_path = output_dir / "candidates.json"
    candidate_path.write_text(
        json.dumps(
            {
                "project_root": str(root),
                "pdf": str(pdf_path),
                "dpi": dpi,
                "figures": candidates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"candidates    {candidate_path}")
    write_gallery(output_dir / "index.html", previews, "Automatic figure candidates")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-extract reviewed figures and propose new crop rectangles."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"crop manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--pdf", type=Path, help="override source PDF")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="render reviewed crops from the manifest"
    )
    extract.add_argument("ids", nargs="*", help="figure IDs, e.g. 2-5 3-1")
    extract.add_argument("--dpi", type=int, help="override manifest DPI")
    extract.add_argument("--force", action="store_true", help="overwrite outputs")
    extract.add_argument(
        "--output-root",
        type=Path,
        help="write relative outputs below this directory",
    )
    extract.add_argument(
        "--gallery",
        type=Path,
        help="write an HTML gallery for the rendered files",
    )
    extract.set_defaults(func=command_extract)

    detect = subparsers.add_parser(
        "detect", help="propose crops above FIGURE x.y captions"
    )
    detect.add_argument(
        "--chapter",
        type=int,
        help="only detect captions belonging to this chapter",
    )
    detect.add_argument("--dpi", type=int, help="candidate output DPI")
    detect.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="candidate images, JSON and gallery destination",
    )
    detect.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="padding around detected content in PDF points",
    )
    detect.add_argument(
        "--vertical-gap",
        type=float,
        default=20.0,
        help="maximum gap joining nearby content in PDF points",
    )
    detect.add_argument(
        "--minimum-top",
        type=float,
        default=54.0,
        help="ignore header content above this PDF y coordinate",
    )
    detect.add_argument(
        "--caption-clearance",
        type=float,
        default=24.0,
        help="minimum boundary below a preceding caption on the same page",
    )
    detect.set_defaults(func=command_detect)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FigureExtractorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except fitz.FileDataError as exc:
        print(f"error: cannot read PDF: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
