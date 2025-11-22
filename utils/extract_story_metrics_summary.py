#!/usr/bin/env python3
"""
Extract per-story metric summaries from ViStoryBench evaluation outputs.

Usage:
  python utils/extract_story_metrics_summary.py
  python utils/extract_story_metrics_summary.py --method TypeMovie --metric cids
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_RESULTS_DIR = REPO_ROOT / "data" / "bench_results"
SUMMARY_ROOT = REPO_ROOT / "data" / "bench_results_summary"

# Mapping between metric directory name and metric fields to persist per story_id.
METRIC_FIELDS = {
    "aesthetic": ["aesthetic_score"],
    "cids": [
        "cids_self_mean",
        "cids_cross_mean",
        "copy_paste_score",
        "occm",
        "single_character_action",
    ],
    "csd": ["self_csd", "cross_csd"],
    "prompt_align": ["scene", "character_action", "camera"],
}

# Table with all (method, mode, language, version, metric, length) rows that should
# be summarized. See user-provided table in the instructions.
RAW_REQUESTS = """
AIbrm Base en 20250819_091354 aesthetic 20
AIbrm Base en 20250819_091354 csd 20
AIbrm Base en 20250819_091354 prompt_align 20
AIbrm Base en 20251031_033329 cids 20
AnimDirector SD3 en 20250819_091354 aesthetic 80
AnimDirector SD3 en 20250819_091354 csd 80
AnimDirector SD3 en 20250819_091354 prompt_align 80
AnimDirector SD3 en 20251030_204809 cids 80
CharaConsist Base en 20251106_214201 aesthetic 80
CharaConsist Base en 20251106_214201 cids 80
CharaConsist Base en 20251106_214201 csd 80
CharaConsist Base en 20251106_214201 prompt_align 16
DouBao Base en 20250819_091355 aesthetic 20
DouBao Base en 20250819_091355 csd 20
DouBao Base en 20250819_091355 prompt_align 20
DouBao Base en 20251101_040214 cids 20
GPT4o Base en 20251111_154843 aesthetic 17
GPT4o Base en 20251111_154843 cids 17
GPT4o Base en 20251111_154843 csd 17
GPT4o Base en 20251111_154843 prompt_align 17
Gemini Base en 20251111_164538 aesthetic 19
Gemini Base en 20251111_164538 cids 19
Gemini Base en 20251111_164538 csd 19
Gemini Base en 20251111_164538 prompt_align 19
MMStoryAgent Base en 20250819_091357 aesthetic 80
MMStoryAgent Base en 20250819_091357 csd 80
MMStoryAgent Base en 20250819_091357 prompt_align 80
MMStoryAgent Base en 20251101_061635 cids 80
MOKI Base en 20250819_091357 aesthetic 20
MOKI Base en 20250819_091357 csd 20
MOKI Base en 20250819_091357 prompt_align 20
MOKI Base en 20251031_025114 cids 20
MorphicStudio Base en 20250819_091357 aesthetic 19
MorphicStudio Base en 20250819_091357 csd 19
MorphicStudio Base en 20250819_091357 prompt_align 19
MorphicStudio Base en 20251101_085953 cids 19
MovieAgent ROICtrl en 20251010_171955 aesthetic 80
MovieAgent ROICtrl en 20251010_171955 csd 80
MovieAgent ROICtrl en 20251010_171955 prompt_align 80
MovieAgent ROICtrl en 20251101_154433 cids 80
MovieAgent SD3 en 20251013_012141 aesthetic 80
MovieAgent SD3 en 20251013_012141 csd 80
MovieAgent SD3 en 20251013_012141 prompt_align 80
MovieAgent SD3 en 20251102_144341 cids 80
NaiveBaseline Base en 20250819_091358 aesthetic 80
NaiveBaseline Base en 20250819_091358 csd 80
NaiveBaseline Base en 20250819_091358 prompt_align 80
NaiveBaseline Base en 20251102_183032 cids 80
NanoBanana Gemini2.5FlashImagePreview en 20251111_174423 aesthetic 20
NanoBanana Gemini2.5FlashImagePreview en 20251111_174423 cids 20
NanoBanana Gemini2.5FlashImagePreview en 20251111_174423 csd 20
NanoBanana Gemini2.5FlashImagePreview en 20251111_174423 prompt_align 14
OmniGen2 Base en 20251111_185620 aesthetic 80
OmniGen2 Base en 20251111_185620 cids 80
OmniGen2 Base en 20251111_185620 csd 80
OmniGen2 Base en 20251111_185620 prompt_align 80
QwenImageEdit2509 base en 20251106_114536 prompt_align 35
QwenImageEdit2509 base en 20251112_103543 aesthetic 80
QwenImageEdit2509 base en 20251112_103543 cids 80
QwenImageEdit2509 base en 20251112_103543 csd 80
SeedStory Base en 20250819_091358 aesthetic 78
SeedStory Base en 20250819_091358 csd 78
SeedStory Base en 20250819_091358 prompt_align 22
SeedStory Base en 20251103_025134 cids 78
Seedream4 base en 20251112_140400 aesthetic 20
Seedream4 base en 20251112_140400 csd 20
Seedream4 base en 20251112_140400 prompt_align 17
Seedream4 base en 20251114_010255 cids 20
ShenBi Base en 20250819_091358 aesthetic 18
ShenBi Base en 20250819_091358 csd 18
ShenBi Base en 20250819_091358 prompt_align 18
ShenBi Base en 20251103_040758 cids 18
Sora2 ALL_ImgRef en 20251114_114833 aesthetic 6
Sora2 ALL_ImgRef en 20251114_114833 cids 6
Sora2 ALL_ImgRef en 20251114_114833 csd 6
Sora2 ALL_ImgRef en 20251114_114833 prompt_align 6
Sora2 ALL_TextOnly en 20251114_120136 aesthetic 15
Sora2 ALL_TextOnly en 20251114_120136 cids 15
Sora2 ALL_TextOnly en 20251114_120136 csd 15
Sora2 ALL_TextOnly en 20251114_120136 prompt_align 15
StoryAdapter ImgRef_Scale0 en 20250819_091400 aesthetic 80
StoryAdapter ImgRef_Scale0 en 20250819_091400 csd 80
StoryAdapter ImgRef_Scale0 en 20250819_091400 prompt_align 80
StoryAdapter ImgRef_Scale0 en 20251103_044933 cids 80
StoryAdapter ImgRef_Scale5 en 20251009_231121 aesthetic 80
StoryAdapter ImgRef_Scale5 en 20251009_231121 csd 80
StoryAdapter ImgRef_Scale5 en 20251009_231121 prompt_align 80
StoryAdapter ImgRef_Scale5 en 20251103_093527 cids 80
StoryAdapter TextOnly_Scale0 en 20251010_035345 aesthetic 80
StoryAdapter TextOnly_Scale0 en 20251010_035345 csd 80
StoryAdapter TextOnly_Scale0 en 20251010_035345 prompt_align 80
StoryAdapter TextOnly_Scale0 en 20251103_125523 cids 80
StoryAdapter TextOnly_Scale5 en 20251010_081735 aesthetic 80
StoryAdapter TextOnly_Scale5 en 20251010_081735 cids 80
StoryAdapter TextOnly_Scale5 en 20251010_081735 csd 80
StoryAdapter TextOnly_Scale5 en 20251010_081735 prompt_align 80
StoryDiffusion ImgRef_Photomaker en 20250819_091407 aesthetic 80
StoryDiffusion ImgRef_Photomaker en 20250819_091407 csd 80
StoryDiffusion ImgRef_Photomaker en 20250819_091407 prompt_align 80
StoryDiffusion ImgRef_Photomaker en 20251031_051447 cids 80
StoryDiffusion Original en 20251010_125027 aesthetic 80
StoryDiffusion Original en 20251010_125027 csd 80
StoryDiffusion Original en 20251010_125027 prompt_align 80
StoryDiffusion Original en 20251031_082051 cids 80
StoryGen AutoRegressive en 20251010_230629 aesthetic 80
StoryGen AutoRegressive en 20251010_230629 csd 80
StoryGen AutoRegressive en 20251010_230629 prompt_align 80
StoryGen AutoRegressive en 20251031_114205 cids 80
StoryGen Mix en 20250819_091410 aesthetic 80
StoryGen Mix en 20250819_091410 csd 80
StoryGen Mix en 20250819_091410 prompt_align 80
StoryGen Mix en 20251031_142625 cids 80
StoryGen MultiImageCondition en 20251011_031116 aesthetic 80
StoryGen MultiImageCondition en 20251011_031116 csd 80
StoryGen MultiImageCondition en 20251011_031116 prompt_align 80
StoryGen MultiImageCondition en 20251101_005055 cids 80
TheaterGen Base en 20250819_091412 aesthetic 80
TheaterGen Base en 20250819_091412 csd 80
TheaterGen Base en 20250819_091412 prompt_align 80
TheaterGen Base en 20251031_002623 cids 80
TypeMovie Base en 20250819_091414 aesthetic 20
TypeMovie Base en 20250819_091414 csd 20
TypeMovie Base en 20250819_091414 prompt_align 20
TypeMovie Base en 20251114_004305 cids 20
TypeMovie Base en 20251114_012657 cids 20
UNO Base en 20250819_091414 aesthetic 80
UNO Base en 20250819_091414 cids 80
UNO Base en 20250819_091414 csd 80
UNO Base en 20250819_091414 prompt_align 80
Vlogger ImgRef en 20250819_091414 aesthetic 80
Vlogger ImgRef en 20250819_091414 cids 80
Vlogger ImgRef en 20250819_091414 csd 80
Vlogger ImgRef en 20250819_091414 prompt_align 80
Vlogger TextOnly en 20251009_145900 aesthetic 80
Vlogger TextOnly en 20251009_145900 cids 80
Vlogger TextOnly en 20251009_145900 csd 80
Vlogger TextOnly en 20251009_145900 prompt_align 80
"""


@dataclass(frozen=True)
class SummaryRequest:
    method: str
    mode: str
    language: str
    version: str
    metric: str
    story_count: int


def parse_requests(raw: str) -> List[SummaryRequest]:
    """Parse whitespace separated rows into SummaryRequest objects."""
    requests: List[SummaryRequest] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) != 6:
            raise ValueError(f"Unexpected row format: {line}")
        method, mode, language, version, metric, story_count = cols
        requests.append(
            SummaryRequest(
                method=method,
                mode=mode,
                language=language,
                version=version,
                metric=metric,
                story_count=int(story_count),
            )
        )
    return requests


def metric_fields(metric_name: str) -> List[str]:
    if metric_name not in METRIC_FIELDS:
        raise KeyError(
            f"Unsupported metric '{metric_name}'. Known metrics: {sorted(METRIC_FIELDS)}"
        )
    return METRIC_FIELDS[metric_name]


def read_story_results(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing story_results.json -> {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_output_name(
    request: SummaryRequest,
    sub_metric: str,
    duplicates_map: Dict[Tuple[str, str, str], int],
) -> str:
    """Return filename stem under method_mode directory."""
    key = (request.method, request.mode, request.metric)
    if duplicates_map.get(key, 0) > 1:
        return f"{sub_metric}_{request.version}"
    return sub_metric


def summarize_request(
    request: SummaryRequest,
    duplicates_map: Dict[Tuple[str, str, str], int],
    dry_run: bool = False,
) -> List[Tuple[str, Path]]:
    metric = request.metric
    metric_dir = (
        BENCH_RESULTS_DIR
        / request.method
        / request.mode
        / request.language
        / request.version
        / "metrics"
        / metric
    )
    story_path = metric_dir / "story_results.json"
    data = read_story_results(story_path)
    if len(data) != request.story_count:
        raise ValueError(
            f"{story_path} has {len(data)} stories but expected {request.story_count}"
        )
    fields = metric_fields(metric)
    summaries: Dict[str, Dict[str, float | int | None]] = {
        field: {} for field in fields
    }
    for story_id in sorted(data.keys()):
        metrics_block = data[story_id].get("metrics", {})
        for field in fields:
            summaries[field][story_id] = metrics_block.get(field)

    out_dir = SUMMARY_ROOT / f"{request.method}_{request.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Tuple[str, Path]] = []
    for field, summary in summaries.items():
        out_name = build_output_name(request, field, duplicates_map)
        out_path = out_dir / f"{out_name}.json"
        if not dry_run:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
        outputs.append((field, out_path))

    return outputs


def filter_requests(
    requests: Iterable[SummaryRequest],
    methods: List[str] | None,
    modes: List[str] | None,
    metrics: List[str] | None,
    versions: List[str] | None,
) -> List[SummaryRequest]:
    selected: List[SummaryRequest] = []
    for req in requests:
        if methods and req.method not in methods:
            continue
        if modes and req.mode not in modes:
            continue
        if metrics and req.metric not in metrics:
            continue
        if versions and req.version not in versions:
            continue
        selected.append(req)
    return selected


def count_duplicates(requests: Iterable[SummaryRequest]) -> Dict[Tuple[str, str, str], int]:
    counts: Dict[Tuple[str, str, str], int] = {}
    for req in requests:
        key = (req.method, req.mode, req.metric)
        counts[key] = counts.get(key, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract metric summaries from story_results.json files using predefined table"
        )
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Filter to specific method(s). Repeat flag to include multiple.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        help="Filter to specific mode(s). Repeat flag to include multiple.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        help="Filter to specific metric(s). Repeat flag to include multiple.",
    )
    parser.add_argument(
        "--version",
        action="append",
        help="Filter to specific timestamp/version(s). Repeat flag to include multiple.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files, only print planned outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests = parse_requests(RAW_REQUESTS)
    requests = filter_requests(
        requests,
        methods=args.method,
        modes=args.mode,
        metrics=args.metric,
        versions=args.version,
    )
    if not requests:
        print("No matching rows; check filters.")
        return

    duplicates = count_duplicates(requests)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    total_outputs = sum(len(metric_fields(req.metric)) for req in requests)
    action = "Would write" if args.dry_run else "Wrote"
    completed = 0
    for req in requests:
        out_paths = summarize_request(req, duplicates, dry_run=args.dry_run)
        for field, out_path in out_paths:
            completed += 1
            print(
                f"[{completed:03d}/{total_outputs:03d}] {action} "
                f"{req.method}/{req.mode}/{req.language}/{req.version} -> "
                f"{req.metric}:{field} ({req.story_count} stories) => "
                f"{out_path.relative_to(REPO_ROOT)}"
            )


if __name__ == "__main__":
    main()
