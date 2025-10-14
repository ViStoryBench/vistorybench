#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage example:
- After starting the local service (see omnigen2/server.py), generate all stories (default en):
    python data_process/inference_custom/omnigen2/run_custom.py --language en --timestamp 20250911_120000
- Specify stories (e.g., 01,02) and override the output root path and service address:
    python data_process/inference_custom/omnigen2/run_custom.py --language ch --story_ids 01,02 --outputs_root data/outputs --server_url http://127.0.0.1:8000 --timestamp 20250911_120000
- Provide service address and API key via environment variables (key is optional):
    export OMNIGEN2_SERVER_URL="http://127.0.0.1:8000"
    export OMNIGEN2_API_KEY="sk-..."  # Passthrough only, no authentication required
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

DEFAULT_METHOD = "omnigen2"
DEFAULT_MODE = "base"
DEFAULT_LANGUAGE = "en"
ENV_SERVER_URL = "OMNIGEN2_SERVER_URL"
ENV_API_KEY = "OMNIGEN2_API_KEY"

NEGATIVE_PROMPT_DEFAULT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

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

def generate_with_omnigen2_server(
    server_url: str,
    api_model: str,
    api_key: Optional[str],
    prompt: str,
    char_image_paths: List[str],
    prev_image_path: Optional[str],
    gen_cfg: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    sleep: float = 2.0,
    timeout: float = 120.0,
) -> Optional[np.ndarray]:
    """
    Call the local OmniGen2 REST service for a single generation.
    Input: prompt, list of character reference image paths, previous shot path (optional), generation configuration
    Output: OpenCV BGR image (np.ndarray) or None
    """
    base = server_url.rstrip("/")
    url = f"{base}/v1beta/models/{api_model}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {}
    if api_key:
        params["key"] = api_key

    # Assemble parts
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for p in (char_image_paths or []):
        b64 = b64_jpeg_from_image_path(p)
        if b64:
            parts.append({"inlineData": {"mime_type": "image/jpeg", "data": b64}})
    if prev_image_path:
        b64_prev = b64_jpeg_from_image_path(prev_image_path)
        if b64_prev:
            parts.append({"type": "text", "text": "last reference shot:"})
            parts.append({"inlineData": {"mime_type": "image/jpeg", "data": b64_prev}})

    payload = {
        "contents": [{
            "role": "user",
            "parts": parts
        }],
        "generationConfig": gen_cfg or {}
    }

    for attempt in range(1, retries + 1):
        try:
            _log(f"[REQ] POST {url} attempt {attempt}/{retries} | parts={len(parts)}")
            resp = requests.post(url, headers=headers, params=params, json=payload, timeout=timeout)
            if not (200 <= resp.status_code < 300):
                _log(f"[WARN] Non-2xx response: {resp.status_code} {resp.text[:200]}")
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise ValueError("Response is missing candidates")
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
                _log(f"[INFO] Call failed: {e} | Retrying in {delay:.1f}s")
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
        _log(f"[ERROR] Exception on save: {out_path} | {e}")
        return False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViStory Multi-shot Generation (OmniGen2 Local REST Service)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Configuration and Paths
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml; defaults to the config.yaml in the repository root")
    parser.add_argument("--dataset_root", type=str, default=None, help="Override the dataset root path")
    parser.add_argument("--outputs_root", type=str, default=None, help="Override the output root path")
    parser.add_argument("--split", type=str, choices=["lite","full"], default="full", help="Select the story list split")
    # Language / Method / Mode / Timestamp
    parser.add_argument("--language", type=str, choices=["ch","en"], default=DEFAULT_LANGUAGE)
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, help="Directory name can contain dots")
    parser.add_argument("--timestamp", type=str, default="20250914_125319", help="YYYYMMDD_HHMMSS; defaults to the current time")
    # Story Selection
    parser.add_argument("--story_ids", type=str, default=None, help="Comma-separated list of story IDs, e.g., 01,02")
    # Service and API Key
    parser.add_argument("--server_url", type=str, default=os.getenv(ENV_SERVER_URL, "http://js2.blockelite.cn:17470"), help=f"OmniGen2 service base URL, can also use environment variable {ENV_SERVER_URL}")
    parser.add_argument("--api_key", type=str, default=os.getenv(ENV_API_KEY, None), help=f"Can use this parameter or environment variable {ENV_API_KEY} (service does not validate by default)")
    # Generation Parameters
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--text_guidance_scale", type=float, default=5.0)
    parser.add_argument("--image_guidance_scale", type=float, default=2.0)
    parser.add_argument("--cfg_range_start", type=float, default=0.0)
    parser.add_argument("--cfg_range_end", type=float, default=1.0)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="euler", choices=["euler","dpmsolver++"])
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT_DEFAULT)
    # Retries
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=2.0, help="Initial retry delay in seconds, with exponential backoff")
    parser.add_argument("--timeout", type=float, default=300, help="HTTP request timeout in seconds")
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
    server_url = args.server_url
    api_key = args.api_key

    api_model = f"{method}-{mode}"

    # Data Loading
    dataset_dir = os.path.join(dataset_root, "ViStory")
    dataset = StoryDataset(dataset_dir)
    all_story_ids = dataset.get_story_name_list(split=args.split)
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
    _log(f"[INFO] Using service: {server_url} | Model: {api_model}")

    # Organize generationConfig
    gen_cfg = {
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "text_guidance_scale": args.text_guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "cfg_range_start": args.cfg_range_start,
        "cfg_range_end": args.cfg_range_end,
        "num_images_per_prompt": args.num_images_per_prompt,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "negative_prompt": args.negative_prompt,
    }

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
        first_path = os.path.join(out_story_dir, f"shot_{0:02d}.png")
        if os.path.isfile(first_path):
            prev_image_path = None

        for idx, shot in enumerate(shots_all):
            out_path = os.path.join(out_story_dir, f"shot_{idx:02d}.png")
            if os.path.isfile(out_path):
                _log(f"[SKIP] Already exists, skipping: {sid} #{idx:02d} -> {out_path}")
                prev_image_path = None
                continue

            prompt = shot.get("prompt", "")
            char_imgs = [p for p in (shot.get("image_paths") or []) if p and os.path.isfile(p)]
            _log(f"[SHOT] {sid} #{idx:02d} | Character images: {len(char_imgs)} | prev: {'Y' if prev_image_path else 'N'}")

            img = generate_with_omnigen2_server(
                server_url=server_url,
                api_model=api_model,
                api_key=api_key,
                prompt=prompt,
                char_image_paths=char_imgs,
                prev_image_path=prev_image_path,
                gen_cfg=gen_cfg,
                retries=args.retries,
                sleep=args.sleep,
                timeout=args.timeout,
            )
            if img is None:
                _log(f"[ERROR] Generation failed, skipping this shot: {sid} #{idx:02d}")
                prev_image_path = None
                continue
            if save_png(img, out_path):
                _log(f"[SAVE] {sid} #{idx:02d} -> {out_path}")
                prev_image_path = None
            else:
                _log(f"[ERROR] Save failed, abandoning as prev for the next frame: {out_path}")
                prev_image_path = None

    _log("All done.")

if __name__ == "__main__":
    main()