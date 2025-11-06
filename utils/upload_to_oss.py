from __future__ import annotations
import os
import sys
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import oss2  # type: ignore
except Exception:
    oss2 = None  # type: ignore

TIMESTAMP_PATTERN = re.compile(r'^\d{8}[_-]\d{6}$')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff')

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS

def guess_outputs_root(base: Path) -> Path:
    """
    Handle cases like /data/outputs/outputs by auto-stepping into 'outputs' if present.
    """
    if (base / "outputs").is_dir():
        return base / "outputs"
    return base

def load_mapping(mapping_file: Optional[Path], maps: Optional[List[str]]) -> Dict[str, str]:
    """
    Load mapping dict from a JSON file and/or inline --map 'local:remote' pairs.
    """
    m: Dict[str, str] = {}
    # inline pairs local:remote
    for pair in maps or []:
        if ':' in pair:
            k, v = pair.split(':', 1)
            m[k.strip()] = v.strip()
    if mapping_file and mapping_file.exists():
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    m.update({str(k): str(v) for k, v in data.items()})
        except Exception as e:
            logging.warning(f"Failed to load mapping file {mapping_file}: {e}")
    return m

def map_segment(seg: str, mapping: Dict[str, str]) -> str:
    return mapping.get(seg, seg)

def iter_immediate_subdirs(p: Path) -> List[Path]:
    return [d for d in p.iterdir() if d.is_dir()]

def discover_en_dirs(method_dir: Path, dataset: Optional[str], lang: str) -> List[Tuple[str, Path, Optional[str]]]:
    """
    Discover language directories and corresponding mode and dataset names under a method directory.
    Returns list of tuples: (mode, en_dir, dataset_name_or_None)
    Tries patterns:
      1) method/mode/dataset/lang
      2) method/mode/lang
      3) method/dataset/lang/mode
      4) method/lang/mode
    """
    found: List[Tuple[str, Path, Optional[str]]] = []
    # Pattern 1 & 2: method/<mode>/(dataset?)/<lang>
    for mode_dir in iter_immediate_subdirs(method_dir):
        mode = mode_dir.name
        # skip language or dataset named dirs incorrectly detected as mode
        if mode in {"en", "ch"}:
            continue
        # dataset layer
        if dataset and (mode_dir / dataset / lang).is_dir():
            found.append((mode, mode_dir / dataset / lang, dataset))
        elif not dataset:
            # try any dataset between mode and lang
            for ds_dir in iter_immediate_subdirs(mode_dir):
                if (ds_dir / lang).is_dir() and ds_dir.name not in {"en", "ch"}:
                    found.append((mode, ds_dir / lang, ds_dir.name))
        # direct lang under mode
        if (mode_dir / lang).is_dir():
            found.append((mode, mode_dir / lang, None))
    # Pattern 3: method/<dataset>/<lang>/<mode>
    if dataset and (method_dir / dataset / lang).is_dir():
        for mode_dir in iter_immediate_subdirs(method_dir / dataset / lang):
            found.append((mode_dir.name, mode_dir, dataset))
    elif not dataset:
        for ds_dir in iter_immediate_subdirs(method_dir):
            if (ds_dir / lang).is_dir() and ds_dir.name not in {"en", "ch"}:
                for mode_dir in iter_immediate_subdirs(ds_dir / lang):
                    found.append((mode_dir.name, mode_dir, ds_dir.name))
    # Pattern 4: method/<lang>/<mode>
    if (method_dir / lang).is_dir():
        for mode_dir in iter_immediate_subdirs(method_dir / lang):
            found.append((mode_dir.name, mode_dir, None))
    # deduplicate by (mode, path)
    dedup = {}
    for mode, en_path, ds in found:
        key = (mode, str(en_path))
        dedup[key] = (mode, en_path, ds)
    return list(dedup.values())

def find_timestamp_dirs_under_en(en_dir: Path) -> List[Path]:
    """
    Return timestamp directories under en-level:
      - en/<timestamp>
      - fallback: en/<story_id>/<timestamp>
    """
    ts_dirs = [d for d in iter_immediate_subdirs(en_dir) if TIMESTAMP_PATTERN.match(d.name)]
    if ts_dirs:
        return ts_dirs
    # fallback: en/<story_id>/<timestamp>
    candidates: List[Path] = []
    for sid_dir in iter_immediate_subdirs(en_dir):
        inner = [d for d in iter_immediate_subdirs(sid_dir) if TIMESTAMP_PATTERN.match(d.name)]
        candidates.extend(inner)
    # unique by name but keep full path of first occurrence
    uniq_by_name: Dict[str, Path] = {}
    for d in candidates:
        uniq_by_name.setdefault(d.name, d)
    return list(uniq_by_name.values())

def count_images_in_shots(ts_dir: Path) -> int:
    """
    Count images under shots for all stories within a timestamp directory.
    If no shots folder under a story, count all images under the story dir.
    """
    count = 0
    for sid_dir in iter_immediate_subdirs(ts_dir):
        if not sid_dir.is_dir():
            continue
        shots_dir = None
        for cand in iter_immediate_subdirs(sid_dir):
            if cand.name.lower() in {"shots", "shot"} or cand.name.endswith("shot") or cand.name.endswith("shots"):
                shots_dir = cand
                break
        base = shots_dir if shots_dir and shots_dir.is_dir() else sid_dir
        for p in base.rglob('*'):
            if is_image(p):
                count += 1
    return count

def select_best_timestamp(en_dir: Path) -> Optional[Path]:
    """
    Select timestamp directory with the most images (tie-breaker by name).
    """
    ts_dirs = find_timestamp_dirs_under_en(en_dir)
    if not ts_dirs:
        return None
    scored: List[Tuple[int, str, Path]] = []
    for d in ts_dirs:
        try:
            n = count_images_in_shots(d)
        except Exception:
            n = 0
        scored.append((n, d.name, d))
    scored.sort(key=lambda x: (x[0], x[1]))  # ascending
    return scored[-1][2]

def create_bucket(endpoint: str, bucket_name: str, ak: str, sk: str, sts_token: Optional[str]=None):
    if oss2 is None:
        raise RuntimeError("oss2 is not installed. Please `pip install oss2`.")
    if sts_token:
        auth = oss2.StsAuth(ak, sk, sts_token)
    else:
        auth = oss2.Auth(ak, sk)
    return oss2.Bucket(auth, endpoint, bucket_name)

def object_exists(bucket, key: str) -> bool:
    try:
        return bucket.object_exists(key)
    except Exception as e:
        logging.debug(f"Exists check failed for {key}: {e}")
        return False

def resumable_upload(bucket, key: str, filename: Path, checkpoint_dir: Path, multipart_threshold: int, part_size: int, skip_existing: bool, retries: int=3) -> bool:
    if skip_existing and object_exists(bucket, key):
        return True
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store = oss2.ResumableStore(root=str(checkpoint_dir))
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries+1):
        try:
            oss2.resumable_upload(bucket, key, str(filename), store=store, multipart_threshold=multipart_threshold, part_size=part_size, num_threads=1)
            return True
        except Exception as e:
            last_exc = e
            logging.warning(f"Upload failed ({attempt}/{retries}) for {filename} -> {key}: {e}")
    if last_exc:
        logging.error(f"Giving up on {filename}: {last_exc}")
    return False

def list_files_to_upload(base_dir: Path, only_shots: bool) -> List[Path]:
    files: List[Path] = []
    if only_shots:
        # collect shots/* under each story
        for sid_dir in iter_immediate_subdirs(base_dir):
            shots = None
            for cand in iter_immediate_subdirs(sid_dir):
                if cand.name.lower() in {"shots", "shot"} or cand.name.endswith("shot") or cand.name.endswith("shots"):
                    shots = cand
                    break
            base = shots if shots and shots.is_dir() else sid_dir
            for p in base.rglob('*'):
                if p.is_file():
                    files.append(p)
    else:
        for p in base_dir.rglob('*'):
            if p.is_file():
                files.append(p)
    return files

def build_remote_key_template(prefix: str,
                              template: str,
                              method: str,
                              mode: str,
                              dataset: Optional[str],
                              lang: str,
                              ts_name: Optional[str],
                              local_file: Path,
                              ts_dir: Path,
                              mapping: Dict[str, str],
                              omit_timestamp: bool=False,
                              rename_timestamp: Optional[str]=None) -> str:
    """
    Build OSS object key based on a template rather than mirroring local paths.
    Placeholders available:
      {method} {mode} {dataset} {lang} {timestamp} {sid} {subpath}
    - sid: the first-level directory under timestamp (e.g., '01')
    - subpath: path under sid (e.g., 'shots/0001.png')
    Empty placeholders (like dataset/timestamp when absent) are omitted during normalization.
    """
    # derive sid and subpath relative to timestamp directory
    rel = local_file.relative_to(ts_dir)
    parts = rel.parts
    sid = parts[0] if len(parts) > 0 else ''
    subparts = list(parts[1:]) if len(parts) > 1 else []

    # apply mapping to each segment
    method_m = map_segment(method, mapping)
    mode_m = map_segment(mode, mapping)
    dataset_m = map_segment(dataset, mapping) if dataset else ''
    lang_m = map_segment(lang, mapping)
    ts_m = ''
    if ts_name and not omit_timestamp:
        ts_seg = rename_timestamp if rename_timestamp else ts_name
        ts_m = map_segment(ts_seg, mapping)
    sid_m = map_segment(sid, mapping) if sid else ''
    sub_mapped = [map_segment(seg, mapping) for seg in subparts]
    subpath = '/'.join(sub_mapped) if sub_mapped else ''

    # fill template
    body_filled = template.format(
        method=method_m,
        mode=mode_m,
        dataset=dataset_m,
        lang=lang_m,
        timestamp=ts_m,
        sid=sid_m,
        subpath=subpath
    )

    # normalize slashes and remove empty segments
    comps = [seg for seg in body_filled.split('/') if seg]
    if prefix:
        prefix_norm = '/'.join([seg for seg in prefix.strip('/').split('/') if seg])
        if prefix_norm:
            comps = [prefix_norm] + comps
    return '/'.join(comps)


def build_remote_key(prefix: str, method: str, mode: str, dataset: Optional[str], lang: str, ts_name: Optional[str], local_file: Path, base_local_dir: Path, mapping: Dict[str, str], omit_timestamp: bool=False, rename_timestamp: Optional[str]=None) -> str:
    """
    Backward-compatibility wrapper using a sensible default template that
    adapts to OSS path rules (no timestamp by default). Prefer using
    build_remote_key_template with --remote-template for full control.
    """
    default_template = "{method}/{mode}/{dataset}/{lang}/{sid}/{subpath}"
    return build_remote_key_template(
        prefix=prefix,
        template=default_template,
        method=method,
        mode=mode,
        dataset=dataset,
        lang=lang,
        ts_name=ts_name,
        local_file=local_file,
        ts_dir=base_local_dir,
        mapping=mapping,
        omit_timestamp=omit_timestamp,
        rename_timestamp=rename_timestamp
    )

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload ViStory outputs to Aliyun OSS with resume support.")
    p.add_argument('--outputs-root', type=str, default='data/outputs', help='Local outputs root (default: data/outputs)')
    p.add_argument('--method', type=str, default=None, help='Filter specific method (e.g., Animdirector)')
    p.add_argument('--mode', type=str, default=None, help='Filter specific mode (e.g., sd3)')
    p.add_argument('--dataset', type=str, default='ViStory', help='Dataset folder name to target (default: ViStory). Use "" to ignore.')
    p.add_argument('--lang', type=str, default='en', help='Language folder name (default: en)')
    p.add_argument('--only-shots', action='store_true', help='Only upload shots folders under each story.')
    p.add_argument('--omit-timestamp', action='store_true', help='Omit timestamp segment in remote path (default: template excludes timestamp).')
    p.add_argument('--rename-timestamp', type=str, default=None, help='Rename timestamp segment in remote path (e.g., latest).')
    p.add_argument('--endpoint', type=str, default=os.getenv('ALIYUN_OSS_ENDPOINT',' oss-cn-shanghai.aliyuncs.com'), help='OSS endpoint (e.g., oss-cn-shanghai.aliyuncs.com)')
    p.add_argument('--bucket', type=str, default=os.getenv('ALIYUN_OSS_BUCKET','bmk-data'), help='OSS bucket name')
    p.add_argument('--ak', type=str, default=os.getenv('ALIYUN_OSS_AK'), help='AccessKey ID (or set env ALIYUN_OSS_AK)')
    p.add_argument('--sk', type=str, default=os.getenv('ALIYUN_OSS_SK'), help='AccessKey Secret (or set env ALIYUN_OSS_SK)')
    p.add_argument('--sts', type=str, default=os.getenv('ALIYUN_OSS_STS_TOKEN'), help='STS token (optional)')
    p.add_argument('--prefix', type=str, default='vistory/outputs', help='Remote OSS prefix (no leading slash).')
    p.add_argument('--remote-template', type=str, default='{method}/{mode}/{dataset}/{lang}/{sid}/{subpath}', help='Remote key template. Placeholders: {method},{mode},{dataset},{lang},{timestamp},{sid},{subpath}. Empty segments are omitted.')
    p.add_argument('--checkpoint-dir', type=str, default='.oss_checkpoints', help='Directory to store resumable upload checkpoints.')
    p.add_argument('--multipart-threshold', type=int, default=5*1024*1024, help='Multipart threshold bytes (default 5MB).')
    p.add_argument('--part-size', type=int, default=5*1024*1024, help='Multipart part size bytes (default 5MB).')
    p.add_argument('--workers', type=int, default=8, help='Concurrent upload workers.')
    p.add_argument('--skip-existing', action='store_true', help='Skip upload if object exists in bucket.')
    p.add_argument('--dry-run', action='store_true', help='Print actions without uploading.')
    p.add_argument('--mapping-file', type=str, default=None, help='JSON mapping file for folder/file name mapping.')
    p.add_argument('--map', action='append', default=None, help='Inline mapping pairs "local:remote". Can be repeated.')
    p.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format='[%(levelname)s] %(message)s')
    outputs_root = guess_outputs_root(Path(args.outputs_root))
    dataset = args.dataset if args.dataset else None
    mapping = load_mapping(Path(args.mapping_file) if args.mapping_file else None, args.map)
    # validate credentials
    endpoint = args.endpoint
    bucket_name = args.bucket
    ak = args.ak
    sk = args.sk
    if not endpoint or not bucket_name or not ak or not sk:
        logging.error("Missing OSS credentials. Provide --endpoint --bucket --ak --sk or set env vars.")
        return 2
    # prepare bucket
    if args.dry_run:
        bucket = None
    else:
        try:
            bucket = create_bucket(endpoint, bucket_name, ak, sk, args.sts)
        except Exception as e:
            logging.error(f"Failed to create OSS bucket client: {e}")
            return 3
    checkpoint_dir = Path(args.checkpoint_dir)
    methods = [d for d in iter_immediate_subdirs(outputs_root)]
    if args.method:
        methods = [d for d in methods if d.name == args.method]
    total_files = 0
    uploaded = 0
    dry_skipped = 0
    failed = 0
    tasks: List[Tuple[Path, str]] = []
    items_to_process: List[Tuple[str, str, Optional[str], Path, Path]] = []  # (method, mode, dataset, en_dir, ts_dir)
    for method_dir in methods:
        en_spots = discover_en_dirs(method_dir, dataset, args.lang)
        if args.mode:
            en_spots = [t for t in en_spots if t[0] == args.mode]
        if not en_spots:
            logging.warning(f"No {args.lang} directory found under method {method_dir.name}")
            continue
        for mode, en_dir, ds in en_spots:
            best_ts = select_best_timestamp(en_dir)
            if not best_ts:
                logging.warning(f"No timestamp found under {en_dir}, fallback to en-level")
                best_ts = en_dir
            items_to_process.append((method_dir.name, mode, ds, en_dir, best_ts))
    # enumerate files and submit uploads
    for method, mode, ds, en_dir, ts_dir in items_to_process:
        base_dir = ts_dir
        files = list_files_to_upload(base_dir, args.only_shots)
        if not files:
            logging.info(f"No files to upload in {base_dir}")
            continue
        total_files += len(files)
        for f in files:
            key = build_remote_key_template(args.prefix, args.remote_template, method, mode, ds, args.lang, ts_dir.name, f, base_dir, mapping, args.omit_timestamp, args.rename_timestamp)
            if args.dry_run:
                logging.info(f"[DRY] {f} -> oss://{bucket_name}/{key}")
                dry_skipped += 1
                continue
            tasks.append((f, key))
    if not args.dry_run and tasks:
        assert 'bucket' in locals() and bucket is not None
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            future_to_task = {ex.submit(resumable_upload, bucket, key, f, checkpoint_dir, args.multipart_threshold, args.part_size, args.skip_existing): (f, key) for (f, key) in tasks}
            for future in as_completed(future_to_task):
                f, key = future_to_task[future]
                ok = False
                try:
                    ok = future.result()
                except Exception as e:
                    logging.error(f"Upload exception for {f}: {e}")
                    ok = False
                if ok:
                    uploaded += 1
                else:
                    failed += 1
    logging.info(f"Done. total={total_files}, uploaded={uploaded}, dry_skipped={dry_skipped}, failed={failed}")
    return 0 if failed == 0 else 4

if __name__ == '__main__':
    raise SystemExit(main())