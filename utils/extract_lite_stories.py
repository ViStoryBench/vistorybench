#!/usr/bin/env python3
"""Pull the ViStory LITE subset out of the AVIF outputs and datasets."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable, Set, Tuple

LITE_DATA = [
    "01", "08", "09", "15", "17", "19", "24", "27", "28", "29", "32", "41", "52", "53",
    "55", "57", "60", "64", "68", "79",
]



def normalize_ids(values: Iterable[str]) -> Set[str]:
    """Normalize ids to two-digit strings and drop empties."""
    normalized: Set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        normalized.add(token.zfill(2))
    return normalized


def parse_lite_ids(value: str) -> Set[str]:
    """Parse a comma/space-separated lite id list."""
    if not value:
        return set()
    tokens = re.split(r"[,\s]+", value.strip())
    return normalize_ids(tokens)


def copy_directory(src: Path, dest: Path, *, overwrite: bool) -> bool:
    """Copy a directory tree, optionally overwriting an existing destination."""
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if overwrite:
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        else:
            return False
    shutil.copytree(src, dest)
    return True


def extract_outputs_lite(
    outputs_root: Path,
    target_root: Path,
    lite_ids: Set[str],
    *,
    overwrite: bool,
) -> Tuple[int, Set[str]]:
    """Copy each lite story directory from all methods/modes into target_root."""
    copied = 0
    found_ids: Set[str] = set()
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"Expected outputs root does not exist: {outputs_root}")
    target_root.mkdir(parents=True, exist_ok=True)

    for method_dir in sorted(outputs_root.iterdir(), key=lambda p: p.name):
        if not method_dir.is_dir():
            continue
        for mode_dir in sorted(method_dir.iterdir(), key=lambda p: p.name):
            if not mode_dir.is_dir():
                continue
            for lang_dir in sorted(mode_dir.iterdir(), key=lambda p: p.name):
                if not lang_dir.is_dir():
                    continue
                for timestamp_dir in sorted(lang_dir.iterdir(), key=lambda p: p.name):
                    if not timestamp_dir.is_dir():
                        continue
                    for story_dir in sorted(timestamp_dir.iterdir(), key=lambda p: p.name):
                        if not story_dir.is_dir():
                            continue
                        story_id = story_dir.name
                        if story_id not in lite_ids:
                            continue
                        found_ids.add(story_id)
                        dest_dir = (
                            target_root
                            / method_dir.name
                            / mode_dir.name
                            / lang_dir.name
                            / timestamp_dir.name
                            / story_id
                        )
                        if copy_directory(story_dir, dest_dir, overwrite=overwrite):
                            copied += 1
    return copied, found_ids


def extract_dataset_lite(
    dataset_root: Path,
    target_root: Path,
    lite_ids: Set[str],
    *,
    overwrite: bool,
) -> Tuple[int, Set[str]]:
    """Copy the lite story directories from dataset_root into target_root."""
    copied = 0
    found_ids: Set[str] = set()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Expected dataset root does not exist: {dataset_root}")
    target_root.mkdir(parents=True, exist_ok=True)

    for entry in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        story_id = entry.name
        if story_id not in lite_ids:
            continue
        found_ids.add(story_id)
        dest_dir = target_root / story_id
        if copy_directory(entry, dest_dir, overwrite=overwrite):
            copied += 1
    return copied, found_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the ViStory-lite story folders out of the AVIF outputs and dataset trees, "
            "preserving the original directory layout."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("data/outputs_avif"),
        help="Root of the AVIF output runs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/dataset_avif"),
        help="Root of the AVIF dataset.",
    )
    parser.add_argument(
        "--lite-outputs-root",
        type=Path,
        default=Path("data/outputs_lite_avif"),
        help="Destination for the lite outputs subset.",
    )
    parser.add_argument(
        "--lite-dataset-root",
        type=Path,
        default=Path("data/dataset_lite_avif"),
        help="Destination for the lite dataset subset.",
    )
    parser.add_argument(
        "--lite-ids",
        type=str,
        default=",".join(LITE_DATA),
        help="Comma/space separated list of two-digit lite story ids.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing lite folders.",
    )
    parser.add_argument(
        "--skip-outputs",
        action="store_true",
        help="Do not gather outputs (only dataset subset).",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Do not gather dataset entries (only outputs subset).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    lite_ids = parse_lite_ids(args.lite_ids)
    if not lite_ids:
        raise SystemExit("No lite story IDs provided.")

    summary: list[str] = []

    if not args.skip_outputs:
        copied_outputs, found_outputs = extract_outputs_lite(
            args.outputs_root,
            args.lite_outputs_root,
            lite_ids,
            overwrite=args.overwrite,
        )
        missing_outputs = sorted(lite_ids - found_outputs)
        summary.append(
            f"Outputs: copied {copied_outputs} story dirs "
            f"({len(found_outputs)} ids found, missing {len(missing_outputs)})"
        )
        if missing_outputs:
            summary.append(f"Missing outputs ids: {', '.join(missing_outputs)}")

    if not args.skip_dataset:
        copied_dataset, found_dataset = extract_dataset_lite(
            args.dataset_root,
            args.lite_dataset_root,
            lite_ids,
            overwrite=args.overwrite,
        )
        missing_dataset = sorted(lite_ids - found_dataset)
        summary.append(
            f"Dataset: copied {copied_dataset} story dirs "
            f"({len(found_dataset)} ids found, missing {len(missing_dataset)})"
        )
        if missing_dataset:
            summary.append(f"Missing dataset ids: {', '.join(missing_dataset)}")

    if summary:
        print("\n".join(summary))
    else:
        print("Nothing to do (both outputs and dataset skipped).")


if __name__ == "__main__":
    main()
