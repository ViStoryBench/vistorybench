#!/usr/bin/env python3
"""Aggregate story counts for benchmark result combinations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count how many stories were evaluated for each method/mode/"
            "language/timestamp/metric combination under data/bench_results."
        )
    )
    parser.add_argument(
        "--root",
        default="data/bench_results",
        help="Root directory that contains per-method benchmark results.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="How to format the output summary (default: table).",
    )
    return parser.parse_args()


def iter_story_result_files(root: Path) -> Iterable[Dict[str, Any]]:
    """Yield metadata for every story_results.json file under the root."""
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mode_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            for lang_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
                for ts_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
                    metrics_dir = ts_dir / "metrics"
                    if not metrics_dir.is_dir():
                        continue
                    for metric_dir in sorted(
                        p for p in metrics_dir.iterdir() if p.is_dir()
                    ):
                        story_file = metric_dir / "story_results.json"
                        if story_file.is_file():
                            yield {
                                "method": method_dir.name,
                                "mode": mode_dir.name,
                                "language": lang_dir.name,
                                "timestamp": ts_dir.name,
                                "metric": metric_dir.name,
                                "path": story_file,
                            }


def count_story_keys(path: Path) -> int:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Story results file {path} does not contain a JSON object")
    story_keys = [key for key in data if key.isdigit()]
    return len(story_keys)


def load_counts(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for meta in iter_story_result_files(root):
        try:
            count = count_story_keys(meta["path"])
        except ValueError as exc:
            print(exc, file=sys.stderr)
            continue
        row = meta.copy()
        row.pop("path")
        row["story_count"] = count
        rows.append(row)
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    columns = ["method", "mode", "language", "timestamp", "metric", "story_count"]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row[col])))
    header = " | ".join(f"{col:<{widths[col]}}" for col in columns)
    divider = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(divider)
    for row in rows:
        print(" | ".join(f"{row[col]:<{widths[col]}}" for col in columns))


def print_csv(rows: List[Dict[str, Any]]) -> None:
    columns = ["method", "mode", "language", "timestamp", "metric", "story_count"]
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root directory {root} not found")
    rows = load_counts(root)
    rows.sort(
        key=lambda row: (
            row["method"],
            row["mode"],
            row["language"],
            row["timestamp"],
            row["metric"],
        )
    )
    if not rows:
        print("No story_results.json files found", file=sys.stderr)
        return
    if args.format == "csv":
        print_csv(rows)
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
