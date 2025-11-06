#!/usr/bin/env python3
import os
import json
import re
import argparse
from datetime import datetime
from pathlib import Path

TIMESTAMP_REGEX = r"^\d{8}[-_]\d{6}$"
TIMESTAMP_RE = re.compile(TIMESTAMP_REGEX)
LANG_CODES = {"en","zh","ch","cn","jp","ja","fr","de","es","ru","ko"}

def find_timestamp_entries(outputs_root: Path):
    """Scan outputs_root and collect all directories whose name matches timestamp format."""
    entries = []
    root_str = str(outputs_root)
    for dirpath, dirnames, filenames in os.walk(root_str):
        base = os.path.basename(dirpath)
        if TIMESTAMP_RE.match(base):
            rel_path = str(Path(dirpath).relative_to(outputs_root).as_posix())
            segments = rel_path.split("/")
            ancestors = segments[:-1]
            method = ancestors[0] if ancestors else None
            language = None
            mode = None
            if len(ancestors) >= 2:
                last = ancestors[-1]
                second_last = ancestors[-2]
                if last in LANG_CODES:
                    language = last
                    mode = second_last
                elif second_last in LANG_CODES:
                    language = second_last
                    mode = last
            entries.append({
                "timestamp": base,
                "path": rel_path,
                "method": method,
                "mode": mode,
                "language": language,
                "ancestors": ancestors,
            })
    return entries

def build_hierarchy(entries):
    """Aggregate timestamps grouped by method/mode/language, de-duplicated."""
    idx = {}
    for e in entries:
        method = e.get("method") or "_unknown"
        mode = e.get("mode") or "_unknown"
        language = e.get("language") or "_unknown"
        ts = e["timestamp"]
        idx.setdefault(method, {}).setdefault(mode, {}).setdefault(language, set()).add(ts)
    # convert sets to sorted lists
    result = {}
    for m, modes in idx.items():
        result[m] = {}
        for md, langs in modes.items():
            result[m][md] = {}
            for lg, ts_set in langs.items():
                result[m][md][lg] = sorted(ts_set)
    return result

def save_index(outputs_root: Path, out_file: Path, entries):
    payload = {
        "root": str(outputs_root),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        # "timestamp_format": "YYYYMMDD[-|_]HHMMSS",
        # "entries": entries,
        "index": build_hierarchy(entries),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Index ViStoryBench outputs up to timestamp level.")
    parser.add_argument("--outputs-root", type=str, default="data/outputs", help="Root of outputs directory (default: data/outputs)")
    parser.add_argument("--out-file", type=str, default="", help="Output JSON file path (default: <outputs-root>/outputs_index.json)")
    args = parser.parse_args()
    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        raise SystemExit(f"Outputs root not found: {outputs_root}")
    out_file = Path(args.out_file) if args.out_file else outputs_root / "outputs_index.json"
    entries = find_timestamp_entries(outputs_root)
    save_index(outputs_root, out_file, entries)
    print(f"[OK] Indexed {len(entries)} timestamp directories -> {out_file}")

if __name__ == "__main__":
    main()