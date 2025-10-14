#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility: Classify ViStory 80 stories into cultural spheres using OpenAI Chat Completions.

Categories:
- Sinosphere (Confucian cultural sphere)
- Japanese cultural sphere
- Indian cultural sphere
- Islamic/Middle Eastern cultural sphere
- Western cultural sphere (can be subdivided)
- Latin American cultural sphere
- African cultural sphere

Reads story.json under data/dataset/ViStory/<ID>/story.json and outputs a CSV/JSONL.

Examples and usage:
  python utils/classify_vistory_culture.py --config config.yaml --out data/result/vistory_culture_classification.csv
  python utils/classify_vistory_culture.py --base_url https://api.openai.com --model gpt-4.1 --api_key sk-xxx
"""
import os
import sys
import json
import argparse
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Prefer using existing GPT utility for HTTP plumbing
from vistorybench.bench.prompt_align.gptv_utils import gptv_query  # type: ignore

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1"
ENV_API_KEY = "VISTORYBENCH_API_KEY"

CATEGORY_DEFINITIONS_EN = {
    "Sinosphere": "Derived from Chinese history, classical literature (e.g., Four Great Classical Novels), wuxia culture, and folk tales.",
    "Japanese": "Derived from Bushido spirit, mono no aware/yuugen aesthetics, yokai lore, and manga culture.",
    "Indian": "Derived from Hindu mythology (e.g., Ramayana, Mahabharata), Bollywood musical paradigm, and caste-related social issues.",
    "Islamic/Middle Eastern": "Derived from One Thousand and One Nights, Islamic stories, Persian epics, and desert nomadic culture.",
    "Western": "Includes Greco-Roman myth, Norse myth, commercial adaptations, cyberpunk, and Eastern Europe/Russia subtypes.",
    "Latin American": "Blends indigenous myths, Catholic culture, and colonial history; magical realism is common.",
    "African": "Derived from tribal legends, oral history, colonial/postcolonial narratives, and African heroic epics.",
}

CATEGORY_LABELS = list(CATEGORY_DEFINITIONS_EN.keys())

SYSTEM_INSTRUCTION_EN = f"""You are a senior cultural studies expert. Given a bilingual (Chinese/English) story text, determine the closest cultural sphere classification and output strict JSON (no extra content).

Cultural sphere classification (must choose one main class + optional subculture note from the options below):
- Sinosphere (Confucian cultural sphere): {CATEGORY_DEFINITIONS_EN["Sinosphere"]}
- Japanese: {CATEGORY_DEFINITIONS_EN["Japanese"]}
- Indian: {CATEGORY_DEFINITIONS_EN["Indian"]}
- Islamic/Middle Eastern: {CATEGORY_DEFINITIONS_EN["Islamic/Middle Eastern"]}
- Western (can be subdivided): {CATEGORY_DEFINITIONS_EN["Western"]}
- Latin American: {CATEGORY_DEFINITIONS_EN["Latin American"]}
- African: {CATEGORY_DEFINITIONS_EN["African"]}

Requirements:
- The main field 'culture' must be one of the English labels above.
- The optional field 'subculture' may refine 'Western' (e.g., "Western–Greco-Roman Myth", "Western–Norse Myth", "Western–Eastern Europe/Russia", "Western–Commercial Adaptation").
- Provide confidence between 0 and 1, and an English rationale (50–120 words).
- If uncertain, choose the closest main class and briefly explain the ambiguity.
- Output a single-line JSON object only; do not include explanations or code fences.

Example JSON:
{{"culture":"Japanese","subculture":"","confidence":0.78,"rationale":"...brief rationale..."}}
"""

def read_yaml_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        if yaml is None:
            # Minimal fallback: attempt JSON then return {}
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            try:
                return json.loads(txt) if txt.startswith("{") else {}
            except Exception:
                return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def resolve_paths_and_api(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str, str, str]:
    # dataset root
    dataset_root = args.dataset_root or (((cfg.get("core") or {}).get("paths") or {}).get("dataset")) or "data/dataset"
    # OpenAI endpoint/model/api_key precedence: CLI > ENV > YAML
    base_url = args.base_url or ((((cfg.get("evaluators") or {}).get("prompt_align") or {}).get("gpt") or {}).get("base_url")) or DEFAULT_BASE_URL
    model = args.model or ((((cfg.get("evaluators") or {}).get("prompt_align") or {}).get("gpt") or {}).get("model")) or DEFAULT_MODEL
    api_key = args.api_key or os.environ.get(ENV_API_KEY) or ((((cfg.get("evaluators") or {}).get("prompt_align") or {}).get("gpt") or {}).get("api_key")) or ""
    return dataset_root, base_url, model, api_key

def list_story_jsons(vistory_dir: Path, select_ids: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    if not vistory_dir.exists():
        return items
    for p in sorted(vistory_dir.iterdir()):
        if not p.is_dir():
            continue
        sid = p.name
        if select_ids and sid not in select_ids and sid.zfill(2) not in select_ids:
            continue
        story_path = p / "story.json"
        if story_path.exists():
            items.append((sid.zfill(2), story_path))
    return items

def read_story_json(path: Path) -> Dict[str, Any]:
    encodings = ["utf-8-sig", "utf-8"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read story JSON: {path} ({last_err})")

def build_story_text(data: Dict[str, Any], prefer_lang: str = "both") -> str:
    def get(d: Dict[str, Any], *keys, default: str = "") -> str:
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur if isinstance(cur, str) else default
    parts: List[str] = []
    story_type_ch = get(data, "Story_type", "ch")
    story_type_en = get(data, "Story_type", "en")
    if story_type_ch or story_type_en:
        parts.append(f"Story type: ch: {story_type_ch} | en: {story_type_en}")
    chars = data.get("Characters") or {}
    if isinstance(chars, dict) and chars:
        parts.append("Characters")
        for name, info in list(chars.items())[:10]:
            name_ch = info.get("name_ch") or ""
            name_en = info.get("name_en") or ""
            prompt_ch = info.get("prompt_ch") or ""
            prompt_en = info.get("prompt_en") or ""
            parts.append(f"- {name_ch or name} / {name_en}: ch: {prompt_ch} | en: {prompt_en}")
    shots = data.get("Shots") or data.get("shots") or []
    if isinstance(shots, list):
        parts.append("Shot summary")
        for s in shots[:30]:
            try:
                idx = s.get("index") or s.get("Index") or ""
                pc_ch = get(s, "Plot Correspondence", "ch")
                pc_en = get(s, "Plot Correspondence", "en")
                sdesc_ch = get(s, "Static Shot Description", "ch")
                sdesc_en = get(s, "Static Shot Description", "en")
                parts.append(f"{idx}. ch: {pc_ch} | en: {pc_en} || ch: {sdesc_ch} | en: {sdesc_en}")
            except Exception:
                continue
    text = "\n".join(parts)
    # Optionally cut to avoid token overflow
    return text[:8000]

def build_messages(story_id: str, story_text: str) -> List[Dict[str, Any]]:
    user_prompt = f"Please classify the following ViStory story (ID: {story_id}) into a cultural sphere:\n\n{story_text}\n\nReturn JSON only."
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION_EN},
        {"role": "user", "content": user_prompt},
    ]

def coerce_json(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    # If wrapped in code fences, strip them
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    # Extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        # Try to fix common trailing commas
        frag2 = re.sub(r",(\s*[}\]])", r"\1", frag)
        try:
            return json.loads(frag2)
        except Exception:
            return None

def classify_one(story_id: str, story_data: Dict[str, Any], model: str, api_key: str, base_url: str, seed: int, top_p: float, temp: float, max_tokens: int, wait_time: int) -> Dict[str, Any]:
    story_text = build_story_text(story_data, "both")
    messages = build_messages(story_id, story_text)
    raw = gptv_query(
        transcript=messages,
        top_p=top_p,
        temp=temp,
        model_type=model,
        api_key=api_key,
        base_url=base_url,
        seed=seed,
        max_tokens=max_tokens,
        wait_time=wait_time,
    )
    parsed = coerce_json(raw) or {}
    out = {
        "story_id": story_id,
        "culture": parsed.get("culture") or "",
        "subculture": parsed.get("subculture") or "",
        "confidence": parsed.get("confidence"),
        "rationale": parsed.get("rationale") or "",
        "raw_response": raw,
    }
    return out

def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["story_id", "culture", "subculture", "confidence", "rationale"]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in headers})

def write_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify ViStory stories into cultural spheres via OpenAI Chat Completions")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--dataset_root", default=None, help="Dataset root directory (default from config.core.paths.dataset or data/dataset)")
    p.add_argument("--vistory_dirname", default="ViStory", help="ViStory dataset directory name under dataset_root")
    p.add_argument("--story_ids", default="", help="Comma-separated story IDs to process (e.g., 01,02,03). Empty = auto discover all.")
    p.add_argument("--out", default="data/result/vistory_culture_classification.csv", help="Output CSV path")
    p.add_argument("--jsonl_out", default="data/result/vistory_culture_classification.jsonl", help="Optional JSONL output path")
    p.add_argument("--base_url", default="https://yunwu.ai", help="OpenAI base URL (e.g., https://api.openai.com or https://api.openai.com/v1)")
    p.add_argument("--model", default="gpt-4.1-mini-2025-04-14", help="OpenAI model id (e.g., gpt-4.1)")
    p.add_argument("--api_key", default="sk-KOC6WE5hmu0pG8HO7dqwd3w61dW7ds4Z3zx1UbSUl9zKqi5H", help=f"API key (default from env {ENV_API_KEY} or config)")
    p.add_argument("--top_p", type=float, default=0.2)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_tokens", type=int, default=384)
    p.add_argument("--wait_time", type=int, default=10, help="Retry wait seconds (gptv_utils)")
    p.add_argument("--sleep", type=float, default=0.6, help="Sleep seconds between API calls")
    p.add_argument("--limit", type=int, default=0, help="Optional: only process first N stories")
    p.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False, help="If true, print prompts but skip API calls")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    cfg = read_yaml_config(args.config)
    dataset_root, base_url, model, api_key = resolve_paths_and_api(cfg, args)
    if not api_key:
        raise SystemExit(f"Missing API key. Provide --api_key or set env {ENV_API_KEY} or put in config.evaluators.prompt_align.gpt.api_key")
    vistory_dir = Path(dataset_root) / (args.vistory_dirname or "ViStory")
    story_ids = [s.strip() for s in args.story_ids.split(",") if s.strip()] if args.story_ids else None
    pairs = list_story_jsons(vistory_dir, story_ids)
    if not pairs:
        raise SystemExit(f"No story.json files found under {vistory_dir}")
    if args.limit and len(pairs) > args.limit:
        pairs = pairs[: args.limit]
    rows: List[Dict[str, Any]] = []
    for i, (sid, spath) in enumerate(pairs, start=1):
        try:
            data = read_story_json(spath)
            if args.dry_run:
                text = build_story_text(data, "both")
                print(f"[DRY] story {sid} prompt preview:\n{(text[:600] + '...') if len(text) > 600 else text}\n")
                rows.append({"story_id": sid, "culture": "", "subculture": "", "confidence": None, "rationale": "", "raw_response": ""})
                continue
            rec = classify_one(
                story_id=sid,
                story_data=data,
                model=model,
                api_key=api_key,
                base_url=base_url,
                seed=args.seed,
                top_p=args.top_p,
                temp=args.temp,
                max_tokens=args.max_tokens,
                wait_time=args.wait_time,
            )
            rows.append(rec)
            print(f"[{i}/{len(pairs)}] {sid} -> {rec.get('culture')} ({rec.get('confidence')})")
            time.sleep(max(0.0, float(args.sleep)))
        except Exception as e:
            print(f"Error on story {sid}: {e}")
            rows.append({"story_id": sid, "culture": "", "subculture": "", "confidence": None, "rationale": "", "raw_response": f"ERROR: {e}"})
    # Write outputs
    write_csv(rows, Path(args.out))
    if args.jsonl_out:
        write_jsonl(rows, Path(args.jsonl_out))
    print(f"Done. CSV written to {args.out}. JSONL: {args.jsonl_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())