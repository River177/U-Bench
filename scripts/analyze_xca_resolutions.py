#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick resolution analysis for the XCA dataset.

Usage:
    python scripts/analyze_xca_resolutions.py \
        --base-dir data/xca_dataset \
        --output-json reports/xca_resolution_stats.json
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_images(base_dir: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (category, file_path) pairs for all images under base_dir."""
    for root, _dirs, files in os.walk(base_dir):
        root_path = Path(root)
        category = "mask" if "ground_truth" in root_path.parts else "image"
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in IMAGE_EXTS:
                yield category, root_path / file


def summarize(resolutions: Counter) -> Dict:
    total = sum(resolutions.values())
    if total == 0:
        return {
            "count": 0,
            "unique_resolutions": 0,
            "min_width": None,
            "max_width": None,
            "min_height": None,
            "max_height": None,
            "top_resolutions": [],
        }

    widths = []
    heights = []
    for (w, h), count in resolutions.items():
        widths.extend([w] * count)
        heights.extend([h] * count)

    top = resolutions.most_common(10)
    return {
        "count": total,
        "unique_resolutions": len(resolutions),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "top_resolutions": [
            {"resolution": f"{w}x{h}", "count": count, "ratio": count / total}
            for (w, h), count in top
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze image resolutions under XCA dataset.")
    parser.add_argument("--base-dir", type=str, default="data/xca_dataset", help="Dataset root directory.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save stats as JSON.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    counters = {"image": Counter(), "mask": Counter()}

    total_files = 0
    for category, path in iter_images(base_dir):
        total_files += 1
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            continue
        counters[category][(width, height)] += 1

    print("=" * 60)
    print(f"Resolution analysis for: {base_dir}")
    print(f"Total image-like files scanned: {total_files}")

    summary = {}
    for category, counter in counters.items():
        stats = summarize(counter)
        summary[category] = stats
        print(f"\n[{category.upper()}]")
        print(f"Count: {stats['count']} (unique resolutions: {stats['unique_resolutions']})")
        if stats["count"] == 0:
            continue
        print(
            f"Width range: {stats['min_width']} - {stats['max_width']}, "
            f"Height range: {stats['min_height']} - {stats['max_height']}"
        )
        print("Top resolutions:")
        for entry in stats["top_resolutions"]:
            res = entry["resolution"]
            count = entry["count"]
            ratio = entry["ratio"] * 100
            print(f"  - {res}: {count} ({ratio:.2f}%)")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved statistics to {output_path}")


if __name__ == "__main__":
    main()
