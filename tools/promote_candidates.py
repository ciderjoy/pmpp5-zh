#!/usr/bin/env python3
"""把已核对的候选图晋升为正式裁剪清单条目。

用法:
    .venv/bin/python tools/promote_candidates.py ch8 8-1 8-2 8-3 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_ROOT = PROJECT_ROOT / "build" / "figure-candidates"
CROPS_PATH = PROJECT_ROOT / "tools" / "figure-crops.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", help="候选图目录名，例如 ch8")
    parser.add_argument("figure_ids", nargs="+")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的条目")
    args = parser.parse_args()

    candidates = json.loads(
        (CANDIDATE_ROOT / args.chapter / "candidates.json").read_text(encoding="utf-8")
    )["figures"]
    by_id = {figure["id"]: figure for figure in candidates}

    manifest = json.loads(CROPS_PATH.read_text(encoding="utf-8"))
    existing = {figure["id"]: index for index, figure in enumerate(manifest["figures"])}

    added, updated, missing = [], [], []
    for figure_id in args.figure_ids:
        source = by_id.get(figure_id)
        if source is None:
            missing.append(figure_id)
            continue
        entry = {
            "id": figure_id,
            "page": source["page"],
            "crop": dict(source["crop"]),
            "output": source["output"],
            "review": "reviewed",
        }
        if figure_id in existing:
            if args.overwrite:
                manifest["figures"][existing[figure_id]] = entry
                updated.append(figure_id)
            else:
                print(f"  [已存在] {figure_id}，如需更新请加 --overwrite")
            continue
        manifest["figures"].append(entry)
        existing[figure_id] = len(manifest["figures"]) - 1
        added.append(figure_id)

    CROPS_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )
    print(f"{args.chapter}: 新增 {len(added)} 条, 更新 {len(updated)} 条")
    if added:
        print("  新增:", ", ".join(added))
    if updated:
        print("  更新:", ", ".join(updated))
    if missing:
        print("  候选清单中缺失:", ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
