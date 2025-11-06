#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility: Augment a stories.json with character names and reference image counts per story.

Input schema (minimal):
{
  "stories": [
    { "id": "01", ... },
    ...
  ]
}

This script reads dataset under config.core.paths.dataset (default data/dataset)/ViStory
and uses StoryDataset.load_characters to extract character display names for en/ch and
count reference images under <story_id>/image/<char_key>/.

Added fields per story:
- characters_en: [ { "name": <str>, "ref_image_count": <int> }, ... ]
- characters_ch: [ { "name": <str>, "ref_image_count": <int> }, ... ]

Usage:
  python utils/augment_stories_with_char_refs.py --input input.json --output output.json
  # Optional overrides:
  python utils/augment_stories_with_char_refs.py --input input.json --dataset-root data/dataset --vistory-dirname ViStory
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root on sys.path so we can import vistorybench.*
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # go to repo root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

from vistorybench.dataset_loader.dataset_load import StoryDataset  # type: ignore

IMG_EXTS = (".png", ".jpg", ".jpeg")

def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    if yaml is None:
        # Minimal fallback: attempt JSON then return {}
        try:
            txt = path.read_text(encoding="utf-8").strip()
            return json.loads(txt) if txt.startswith("{") else {}
        except Exception:
            return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def resolve_dataset_root(cfg: Dict[str, Any], cli_dataset_root: str | None) -> Path:
    core = (cfg or {}).get("core") or {}
    paths = core.get("paths") or {}
    ds_root = cli_dataset_root or paths.get("dataset") or (cfg or {}).get("dataset_path") or "data/dataset"
    return Path(str(ds_root))

def read_json_any(path: Path) -> Dict[str, Any]:
    encs = ["utf-8-sig", "utf-8"]
    last_err: Exception | None = None
    for enc in encs:
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read JSON: {path} ({last_err})")

def write_json_any(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def discover_stories(ds: StoryDataset) -> List[str]:
    try:
        return [str(s).zfill(2) for s in ds.get_story_name_list(split="full")]
    except Exception:
        return []

def build_char_lists(ds: StoryDataset, story_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (characters_en, characters_ch).
    """
    characters_en: List[Dict[str, Any]] = []
    characters_ch: List[Dict[str, Any]] = []
    try:
        en_map = ds.load_characters(story_id, "en")  # key -> {name, key, prompt, images, ...}
    except Exception:
        en_map = {}
    try:
        ch_map = ds.load_characters(story_id, "ch")
    except Exception:
        ch_map = {}

    # Iterate by union of keys (preserve deterministic order by sorted)
    all_keys = sorted(set(list(en_map.keys()) + list(ch_map.keys())))
    for ck in all_keys:
        en_info = en_map.get(ck) or {}
        ch_info = ch_map.get(ck) or {}
        # image list is language-agnostic; prefer en_info.images then ch_info.images
        imgs = en_info.get("images") or ch_info.get("images") or []
        cnt = 0
        # Count only image-like paths (defensive)
        for p in imgs:
            if isinstance(p, str) and p.lower().endswith(IMG_EXTS):
                cnt += 1
        # Names
        en_name = (en_info.get("name") or en_info.get("key") or ck or "").strip()
        ch_name = (ch_info.get("name") or ck or "").strip()
        characters_en.append({"name": en_name, "ref_image_count": int(cnt)})
        characters_ch.append({"name": ch_name, "ref_image_count": int(cnt)})
    return characters_en, characters_ch

def augment(stories_obj: Dict[str, Any], ds: StoryDataset) -> Dict[str, Any]:
    if not isinstance(stories_obj, dict) or "stories" not in stories_obj:
        raise ValueError("Input JSON must be an object with a top-level 'stories' array.")
    items = stories_obj.get("stories")
    if not isinstance(items, list):
        raise ValueError("The 'stories' field must be a list.")
    out_items: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sid_raw = it.get("id")
        sid = str(sid_raw).zfill(2) if sid_raw is not None else ""
        if not sid:
            print(f"Warning: story item without id, skipping.")
            out_items.append(it)
            continue
        try:
            characters_en, characters_ch = build_char_lists(ds, sid)
        except Exception as e:
            print(f"Warning: failed to build characters for story {sid}: {e}")
            characters_en, characters_ch = [], []
        new_rec = dict(it)
        new_rec["characters_en"] = characters_en
        new_rec["characters_ch"] = characters_ch
        out_items.append(new_rec)
    return {"stories": out_items}

def default_output_path(in_path: Path) -> Path:
    if in_path.suffix.lower() == ".json":
        return in_path.with_name(in_path.stem + ".augmented.json")
    return in_path.with_name(in_path.name + ".augmented.json")

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment stories.json with character names and reference image counts per story")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--dataset-root", dest="dataset_root", default='/root/vistorybench/data/dataset', help="Dataset root directory (default from config.core.paths.dataset or data/dataset)")
    p.add_argument("--vistory-dirname", dest="vistory_dirname", default="ViStory", help="ViStory dataset directory name under dataset_root")
    p.add_argument("--input", default='story_info.json', help="Path to input JSON file with top-level 'stories' array")
    p.add_argument("--output", default='story_info.json', help="Path to write augmented JSON (default: alongside input with .augmented.json)")
    return p.parse_args(argv)

def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml_config(Path(args.config))
    dataset_root = resolve_dataset_root(cfg, args.dataset_root)
    vistory_dir = dataset_root / (args.vistory_dirname or "ViStory")
    if not vistory_dir.exists():
        print(f"Warning: ViStory directory not found at: {vistory_dir}")
    ds = StoryDataset(str(vistory_dir))

    in_path = Path(args.input)
    stories_obj = read_json_any(in_path)
    out_obj = augment(stories_obj, ds)

    out_path = Path(args.output) if args.output else default_output_path(in_path)
    write_json_any(out_path, out_obj)
    print(f"Done. Augmented JSON written to: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())