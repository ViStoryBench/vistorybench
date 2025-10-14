#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shutil
import argparse
from typing import Optional, Tuple, Dict, List

TS_PATTERN = re.compile(r'^\d{8}_\d{6}$')

def read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'[WARN] summary not found: {path}')
        return None
    except json.JSONDecodeError as e:
        print(f'[ERROR] invalid JSON in {path}: {e}')
        return None

def write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write('\n')

def classify_run(summary: Dict) -> str:
    metrics = summary.get('metrics', {})
    if not isinstance(metrics, dict):
        return 'unknown'
    keys = set(metrics.keys())
    if keys == {'cids'}:
        return 'cids_only'
    if 'cids' in keys and len(keys) > 1:
        return 'full'
    return 'unknown'

def find_run_pair(language_dir: str) -> Tuple[Optional[str], Optional[str]]:
    ts_dirs = [d for d in os.listdir(language_dir)
               if os.path.isdir(os.path.join(language_dir, d)) and TS_PATTERN.match(d)]
    cids_only_dir = None
    full_dir = None
    for d in sorted(ts_dirs):
        s_path = os.path.join(language_dir, d, 'summary.json')
        s = read_json(s_path)
        if not s:
            continue
        tag = classify_run(s)
        if tag == 'cids_only':
            cids_only_dir = os.path.join(language_dir, d)
        elif tag == 'full':
            full_dir = os.path.join(language_dir, d)
    return cids_only_dir, full_dir

def copy_if_exists(src: str, dst: str, dry_run: bool=False) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if dry_run:
        print(f'[DRY] copy {src} -> {dst}')
    else:
        shutil.copy2(src, dst)
        print(f'[OK ] copied {src} -> {dst}')
    return True

def merge_cids(src_dir: str, dst_dir: str, dry_run: bool=False, backup: bool=True, copy_mid_results: bool=False) -> bool:
    # dataset-level metrics in summary.json
    src_sum_path = os.path.join(src_dir, 'summary.json')
    dst_sum_path = os.path.join(dst_dir, 'summary.json')
    src_summary = read_json(src_sum_path)
    dst_summary = read_json(dst_sum_path)
    if not src_summary or not dst_summary:
        print(f'[WARN] skip merge for {src_dir} -> {dst_dir} due to missing summary.json')
        return False
    src_cids = src_summary.get('metrics', {}).get('cids', {}).get('metrics', {})
    if not src_cids:
        print(f'[WARN] src cids metrics empty: {src_sum_path}')
    # backup
    if backup and not dry_run and os.path.exists(dst_sum_path):
        shutil.copy2(dst_sum_path, dst_sum_path + '.bak')
        print(f'[OK ] backup {dst_sum_path} -> {dst_sum_path}.bak')
    # write merged summary
    if 'metrics' not in dst_summary or not isinstance(dst_summary['metrics'], dict):
        dst_summary['metrics'] = {}
    dst_summary['metrics']['cids'] = dst_summary['metrics'].get('cids', {})
    dst_summary['metrics']['cids']['metrics'] = src_cids
    if dry_run:
        print(f'[DRY] update cids metrics in {dst_sum_path} with values from {src_sum_path}')
    else:
        write_json(dst_sum_path, dst_summary)
        print(f'[OK ] updated cids metrics in {dst_sum_path}')
    # item-level and story-level files
    cids_dir_src = os.path.join(src_dir, 'metrics', 'cids')
    cids_dir_dst = os.path.join(dst_dir, 'metrics', 'cids')
    os.makedirs(cids_dir_dst, exist_ok=True) if not dry_run else None
    copied_any = False
    copied_any |= copy_if_exists(os.path.join(cids_dir_src, 'items.jsonl'),
                                 os.path.join(cids_dir_dst, 'items.jsonl'), dry_run=dry_run)
    copied_any |= copy_if_exists(os.path.join(cids_dir_src, 'story_results.json'),
                                 os.path.join(cids_dir_dst, 'story_results.json'), dry_run=dry_run)
    # optional: scores.json if exists
    copied_any |= copy_if_exists(os.path.join(cids_dir_src, 'scores.json'),
                                 os.path.join(cids_dir_dst, 'scores.json'), dry_run=dry_run)
    # optional mid_results
    if copy_mid_results:
        src_mid = os.path.join(cids_dir_src, 'mid_results')
        dst_mid = os.path.join(cids_dir_dst, 'mid_results')
        if os.path.exists(src_mid):
            if dry_run:
                print(f'[DRY] copy tree {src_mid} -> {dst_mid}')
            else:
                if os.path.exists(dst_mid):
                    shutil.rmtree(dst_mid)
                shutil.copytree(src_mid, dst_mid)
                print(f'[OK ] copied mid_results {src_mid} -> {dst_mid}')
            copied_any = True
    return True

def process(root: str, method: Optional[str]=None, mode: Optional[str]=None, language: Optional[str]=None,
            dry_run: bool=False, backup: bool=True, copy_mid_results: bool=False) -> None:
    methods = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for m in methods:
        if method and m != method:
            continue
        method_dir = os.path.join(root, m)
        modes = [d for d in os.listdir(method_dir) if os.path.isdir(os.path.join(method_dir, d))]
        for md in modes:
            if mode and md != mode:
                continue
            mode_dir = os.path.join(method_dir, md)
            langs = [d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))]
            for lg in langs:
                if language and lg != language:
                    continue
                lang_dir = os.path.join(mode_dir, lg)
                src_dir, dst_dir = find_run_pair(lang_dir)
                if not src_dir or not dst_dir:
                    print(f'[SKIP] {method_dir}/{md}/{lg}: pair not found (src:{src_dir}, dst:{dst_dir})')
                    continue
                print(f'[MERGE] method={m} mode={md} lang={lg} src={os.path.basename(src_dir)} -> dst={os.path.basename(dst_dir)}')
                merged = merge_cids(src_dir, dst_dir, dry_run=dry_run, backup=backup, copy_mid_results=copy_mid_results)
                if not merged:
                    print(f'[FAIL] merge {src_dir} -> {dst_dir}')
                else:
                    print(f'[DONE] merge {src_dir} -> {dst_dir}')

def main():
    parser = argparse.ArgumentParser(description='Overwrite CIDS results (item, story, dataset) from cids-only runs into full runs.')
    parser.add_argument('--root', default='data/bench_results', help='Root path to bench results')
    parser.add_argument('--method', default=None, help='Filter by method name (e.g., bairimeng_ai)')
    parser.add_argument('--mode', default=None, help='Filter by mode name (e.g., base)')
    parser.add_argument('--language', default=None, help='Filter by language (e.g., en)')
    parser.add_argument('--dry-run', action='store_true', help='Preview actions without writing files')
    parser.add_argument('--no-backup', dest='backup', action='store_false', help='Disable summary.json backup before overwrite')
    parser.add_argument('--copy-mid-results', action='store_true', help='Also copy mid_results images for CIDS')
    args = parser.parse_args()
    process(root=args.root, method=args.method, mode=args.mode, language=args.language,
            dry_run=args.dry_run, backup=args.backup, copy_mid_results=args.copy_mid_results)

if __name__ == '__main__':
    main()