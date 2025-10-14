#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage example:
- Generate all stories (default ch):
    python data_process/inference_custom/gemini_vistory/run_custom.py --language ch --timestamp 20250902_120000
- Specify stories (e.g., 01,02) and override the output root path:
    python data_process/inference_custom/gemini_vistory/run_custom.py --language en --story_ids 01,02 --outputs_root data/outputs --timestamp 20250902_120000
- Provide API key via environment variable:
    export YUNWU_API_KEY="sk-..." then run the script
"""

import os
import sys
import argparse
import time
import base64
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

import yaml
import requests
import numpy as np
import cv2

from vistorybench.dataset_loader.dataset_load import StoryDataset

DEFAULT_METHOD = "nano_banana"
DEFAULT_MODE = "gemini-2.5-flash-image-preview"
DEFAULT_LANGUAGE = "en"
ENV_API_KEY = "sk-Uy8ckzV1YDmyZlYPJgkUgyjFN7kEYNafHB2cb8lWFGFuXmsZ"

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_root_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Read the config.yaml from the repository root.
    """
    if config_path and os.path.isfile(config_path):
        cfg_path = Path(config_path)
    else:
        # repo_root is the third-level parent directory of run_custom.py
        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def resolve_paths(args, cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Parse dataset_root and outputs_root based on config.yaml and CLI overrides.
    """
    # repo_root for making relative paths absolute
    repo_root = Path(__file__).resolve().parents[3]
    ds_cfg = (((cfg or {}).get("core") or {}).get("paths") or {}).get("dataset") or "data/dataset"
    out_cfg = (((cfg or {}).get("core") or {}).get("paths") or {}).get("outputs") or "data/outputs"
    dataset_root = args.dataset_root or str((repo_root / ds_cfg).resolve())
    outputs_root = args.outputs_root or str((repo_root / out_cfg).resolve())
    return {"dataset_root": dataset_root, "outputs_root": outputs_root}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def b64_jpeg_from_image_path(img_path: str) -> Optional[str]:
    """
    Read an image using cv2 and encode it as a JPEG base64 string.
    """
    try:
        if not img_path or not os.path.isfile(img_path):
            return None
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception as e:
        _log(f"[WARN] Failed to read/encode image: {img_path} | {e}")
        return None

def decode_image_from_base64(data_b64: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(data_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        _log(f"[WARN] Failed to decode returned image: {e}")
        return None

def generate_with_gemini(
    api_model: str,
    api_key: str,
    prompt: str,
    char_image_paths: List[str],
    prev_image_path: Optional[str],
    retries: int = 3,
    sleep: float = 2.0,
    timeout: float = 90.0
) -> Optional[np.ndarray]:
    """
    Call the yunwu Gemini API for a single generation.
    Input: prompt, list of character initial image paths, previous shot path (optional)
    Output: OpenCV BGR image (np.ndarray) or None
    """
    url = f"https://yunwu.ai/v1beta/models/{api_model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    # Assemble parts
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    # Only the first reference image for each character; skip if no image
    for p in (char_image_paths or []):
        b64 = b64_jpeg_from_image_path(p)
        if b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
    # Previous shot image
    if prev_image_path:
        b64_prev = b64_jpeg_from_image_path(prev_image_path)
        if b64_prev:
            parts.append({"type": "text", "text": "last reference shot:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_prev}})

    payload = {
        "contents": [{
            "role": "user",
            "parts": parts
        }],
        "generationConfig": {"responseModalities": ["TEXT","IMAGE"]}
    }

    for attempt in range(1, retries + 1):
        try:
            _log(f"[REQ] Calling {api_model}, attempt {attempt}/{retries}, parts={len(parts)}")
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if not (200 <= resp.status_code < 300):
                _log(f"[WARN] Non-2xx response: {resp.status_code} {resp.text[:200]}")
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json()
            # Parse the first item containing inlineData
            candidates = data.get("candidates") or []
            if not candidates:
                raise ValueError("Response missing candidates")
            content = candidates[0].get("content") or {}
            parts_out = content.get("parts") or []
            img_b64 = None
            for part in parts_out:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and "data" in inline:
                    img_b64 = inline["data"]
                    break
            if not img_b64:
                raise ValueError("Returned image data inlineData not found")
            img = decode_image_from_base64(img_b64)
            if img is None:
                raise ValueError("Failed to decode returned image")
            return img
        except Exception as e:
            if attempt < retries:
                delay = sleep * (2 ** (attempt - 1))
                _log(f"[INFO] Call failed: {e} | Retrying after {delay:.1f}s")
                time.sleep(delay)
            else:
                _log(f"[ERROR] Multiple failures, giving up: {e}")
                return None

def save_png(img: np.ndarray, out_path: str) -> bool:
    ensure_dir(os.path.dirname(out_path))
    try:
        ok = cv2.imwrite(out_path, img)
        if not ok:
            _log(f"[ERROR] Save failed: {out_path}")
        return ok
    except Exception as e:
        _log(f"[ERROR] Exception during save: {out_path} | {e}")
        return False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViStory multi-shot generation (yunwu Gemini)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Configuration and paths
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml; defaults to repository root config.yaml")
    parser.add_argument("--dataset_root", type=str, default=None, help="Override dataset root path")
    parser.add_argument("--outputs_root", type=str, default=None, help="Override output root path")
    # Language / Method / Mode / Timestamp
    parser.add_argument("--language", type=str, choices=["ch","en"], default=DEFAULT_LANGUAGE)
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, help="Directory name can contain dots")
    parser.add_argument("--timestamp", type=str, default="20250908_152837", help="YYYYMMDD_HHMMSS; defaults to current time")
    # Story selection
    parser.add_argument("--story_ids", type=str, default=None, help="Comma-separated list of story IDs, e.g., 01,02")
    # API Key
    parser.add_argument("--api_key", type=str, default=None, help="Can use this parameter or the environment variable YUNWU_API_KEY")
    # Retries
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=2.0, help="Initial retry delay in seconds, with exponential backoff")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout in seconds")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = load_root_config(args.config)
    paths = resolve_paths(args, cfg)
    dataset_root = paths["dataset_root"]
    outputs_root = paths["outputs_root"]

    language = args.language
    method = args.method
    mode = args.mode
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    api_key = ENV_API_KEY or args.api_key
    if not api_key:
        _log(f"[FATAL] Missing API Key, please set the environment variable {ENV_API_KEY} or pass --api_key")
        sys.exit(1)

    api_model = f"{method}-{mode}"

    # Data loading
    dataset_dir = os.path.join(dataset_root, "ViStory")
    dataset = StoryDataset(dataset_dir)
    all_story_ids = dataset.get_story_name_list(split='lite')
    if not all_story_ids:
        _log(f"[FATAL] Story directory not found: {dataset_dir}")
        sys.exit(1)

    if args.story_ids:
        wanted = [s.strip() for s in args.story_ids.split(",") if s.strip()]
        story_ids = [s for s in wanted if s in all_story_ids]
        missing = sorted(set(wanted) - set(story_ids))
        if missing:
            _log(f"[WARN] The following story IDs do not exist and will be skipped: {','.join(missing)}")
    else:
        story_ids = all_story_ids

    stories = dataset.load_stories(story_ids, language)

    _log(f"[INFO] Dataset: {dataset_dir} | Output: {outputs_root}")
    _log(f"[INFO] method/mode/lang/ts: {method}/{mode}/{language}/{timestamp}")
    count=1
    for sid in story_ids:
        story = stories.get(sid)
        if not story:
            _log(f"[WARN] Skipping empty story: {sid}")
            continue
        shots_all = dataset.story_prompt_merge(story, mode='all')
        out_story_dir = os.path.join(outputs_root, method, mode, language, timestamp, sid)
        ensure_dir(out_story_dir)
        _log(f"[INFO] Starting story: {sid} | {len(shots_all)} shots | Output directory: {out_story_dir}")

        prev_image_path: Optional[str] = None
        # If the first frame already exists, prev_image_path points to it for continuation
        first_path = os.path.join(out_story_dir, f"shot_{0:02d}.png")
        if os.path.isfile(first_path):
            prev_image_path = first_path

        for idx, shot in enumerate(shots_all):
            count += 1
            out_path = os.path.join(out_story_dir, f"shot_{idx:02d}.png")
            if os.path.isfile(out_path):
                _log(f"[SKIP] Already exists, skipping: {sid} #{idx:02d} -> {out_path}")
                prev_image_path = out_path  # Resume from breakpoint: use as the next frame's prev
                continue

            prompt = shot.get("prompt", "")
            char_imgs = [p for p in (shot.get("image_paths") or []) if p and os.path.isfile(p)]
            _log(f"[SHOT] {sid} #{idx:02d} | Character images: {len(char_imgs)} | prev: {'Y' if prev_image_path else 'N'}")

            img = generate_with_gemini(
                api_model=mode,
                api_key=api_key,
                prompt=prompt,
                char_image_paths=char_imgs,
                prev_image_path=prev_image_path,
                retries=args.retries,
                sleep=args.sleep,
                timeout=args.timeout
            )
            if img is None:
                _log(f"[ERROR] Generation failed, skipping this shot: {sid} #{idx:02d}")
                # If it fails, do not pass prev to the next frame to avoid image sequence errors
                prev_image_path = None
                continue
            if save_png(img, out_path):
                _log(f"[SAVE] {sid} #{idx:02d} -> {out_path}")
                prev_image_path = out_path
            else:
                _log(f"[ERROR] Save failed, abandoning as next frame's prev: {out_path}")
                prev_image_path = None
    print("All done.",count)

if __name__ == "__main__":
    main()