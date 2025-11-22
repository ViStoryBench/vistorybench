#!/usr/bin/env python3
from __future__ import annotations

import cmd
import os
import sys
import argparse
import logging
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple, Dict, Iterable, Set
import fnmatch
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
import hashlib
import mimetypes
from tqdm import tqdm  # type: ignore
from dataclasses import dataclass

# Optional dotenv support
from dotenv import load_dotenv  # type: ignore


try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound  # type: ignore
    from botocore.config import Config as BotoConfig  # type: ignore
    from boto3.s3.transfer import TransferConfig  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    boto3 = None  # type: ignore
    ClientError = NoCredentialsError = ProfileNotFound = None  # type: ignore
    BotoConfig = None  # type: ignore
    TransferConfig = None  # type: ignore

# Env alias helper
def get_env_with_aliases(names: List[str], default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

def manual_load_dotenv(dotenv_path: str = ".env") -> bool:
    """
    Minimal .env loader when python-dotenv is unavailable.
    Reads KEY=VALUE pairs (ignores comments/empty lines) and sets os.environ if not already present.
    """
    try:
        p = Path(dotenv_path)
        if not p.exists():
            return False
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        return True
    except Exception as e:
        logging.warning(f"Failed to manually load .env: {e}")
        return False


@dataclass
class Stats:
    scanned: int = 0
    planned: int = 0
    uploaded: int = 0
    skipped: int = 0
    failed: int = 0
    deleted: int = 0
    dry_run: bool = False


def normalize_prefix(prefix: str) -> str:
    if prefix is None:
        return ""
    p = "/".join([seg for seg in str(prefix).strip().replace("\\", "/").split("/") if seg])
    return p


def _parse_src_prefix_pair_raw(value: str) -> Tuple[str, str]:
    if not value:
        raise ValueError("Expected SRC=PREFIX expression, got empty string.")
    separators = ("=>", "::", "=", ":")
    for sep in separators:
        if sep in value:
            left, right = value.split(sep, 1)
            break
    else:
        raise ValueError(f"Invalid SRC=PREFIX expression: '{value}'")

    src = left.strip()
    prefix = right.strip()
    if not src or not prefix:
        raise ValueError(f"Invalid SRC=PREFIX expression (missing value): '{value}'")
    return src, prefix


def parse_src_prefix_pair(value: str) -> Tuple[str, str]:
    """
    Adapter for argparse that surfaces friendly errors.
    """
    try:
        return _parse_src_prefix_pair_raw(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def parse_src_prefix_blob(blob: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not blob:
        return pairs
    normalized = blob.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace(",", ";")
    for entry in normalized.replace("\n", ";").split(";"):
        token = entry.strip()
        if not token:
            continue
        pairs.append(_parse_src_prefix_pair_raw(token))
    return pairs


def to_posix_rel(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    return rel.as_posix()


def should_include(rel_posix: str, includes: List[str], excludes: List[str]) -> bool:
    # Normalize rel_posix for matching
    target = rel_posix
    included = True
    if includes:
        included = any(fnmatch.fnmatch(target, pat) for pat in includes)
    if not included:
        return False
    if excludes:
        if any(fnmatch.fnmatch(target, pat) for pat in excludes):
            return False
    return True


def normalize_skip_dirs(skip_dirs: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for entry in skip_dirs or []:
        if not entry:
            continue
        value = str(entry).strip()
        if not value:
            continue
        value = value.strip("/\\")
        if not value:
            continue
        name = Path(value).name
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def path_has_skip_dir(rel_path: Path, skip_dirs: Set[str]) -> bool:
    if not skip_dirs:
        return False
    dirs = rel_path.parts[:-1]
    if not dirs:
        return False
    for part in dirs:
        if part in skip_dirs:
            return True
    return False


def rel_path_has_skip_dir(rel: str, skip_dirs: Set[str]) -> bool:
    if not skip_dirs or not rel:
        return False
    rel_clean = rel.strip("/")
    if not rel_clean:
        return False
    parts = PurePosixPath(rel_clean).parts
    dirs = parts[:-1]
    if not dirs:
        return False
    return any(part in skip_dirs for part in dirs)


def build_skip_globs(skip_dirs: Iterable[str]) -> List[str]:
    globs: List[str] = []
    seen: Set[str] = set()
    for name in skip_dirs or []:
        if not name:
            continue
        cleaned = str(name).strip()
        if not cleaned:
            continue
        for pattern in (f"{cleaned}/*", f"*/{cleaned}/*"):
            if pattern in seen:
                continue
            seen.add(pattern)
            globs.append(pattern)
    return globs


def detect_aws_cli() -> bool:
    return shutil.which("aws") is not None


def build_aws_cli_sync_command(
    direction: str,
    local: Path,
    bucket: str,
    prefix: str,
    region: Optional[str],
    profile: Optional[str],
    includes: List[str],
    excludes: List[str],
    delete: bool,
    dry_run: bool,
    endpoint_url: Optional[str] = None,
    acl: Optional[str] = None,
    skip_dirs: Iterable[str] = (),
) -> List[str]:
    prefix_norm = normalize_prefix(prefix)
    s3_uri = f"s3://{bucket}"
    if prefix_norm:
        s3_uri = f"{s3_uri}/{prefix_norm}"
    if direction == "download":
        src = s3_uri
        dest = str(local)
    else:
        src = str(local)
        dest = s3_uri
    cmd: List[str] = ["aws", "s3", "sync", src, dest]
    if region:
        cmd += ["--region", region]
    if profile:
        cmd += ["--profile", profile]
    if endpoint_url:
        cmd += ["--endpoint-url", endpoint_url]
    # Only meaningful for uploads
    if acl and direction == "upload":
        cmd += ["--acl", acl]
    # Apply filters. With includes present, start with exclude '*' then include patterns,
    # and then allow user-specified excludes to override.
    if includes:
        cmd += ["--exclude", "*"]
        for pat in includes:
            cmd += ["--include", pat]
        for pat in excludes:
            cmd += ["--exclude", pat]
    else:
        for pat in excludes:
            cmd += ["--exclude", pat]

    for pat in build_skip_globs(skip_dirs):
        cmd += ["--exclude", pat]
    if delete:
        cmd.append("--delete")
    if dry_run:
        cmd.append("--dryrun")
    return cmd


def compute_md5(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def create_boto3_client(profile: Optional[str], region: Optional[str], endpoint_url: Optional[str], access_key: Optional[str], secret_key: Optional[str]):
    """
    Create an S3 client with graceful fallback:
    - If explicit credentials are provided via env/.env, ignore profile.
    - If the requested profile is missing, fall back to no-profile session.
    - If session init fails, fall back to direct boto3.client.
    """
    if boto3 is None:
        return None

    # Prepare botocore config
    try:
        config = BotoConfig(
            region_name=region,
            retries={"max_attempts": 8, "mode": "standard"},
        )
    except Exception:
        # Fallback without config if region is non-standard
        config = None  # type: ignore

    # Decide whether to honor profile
    use_profile = bool(profile) and not (access_key or secret_key)

    # Try to build session
    session = None
    try:
        if use_profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
    except Exception as e:
        # Specifically handle missing profile when available
        if ProfileNotFound and isinstance(e, ProfileNotFound):
            logging.warning(f"AWS profile '{profile}' not found, falling back to environment credentials.")
        else:
            logging.warning(f"Failed to initialize boto3 Session: {e}. Falling back to direct client.")
        session = None

    # Try client from session
    if session is not None:
        try:
            client = session.client(
                "s3",
                config=config,
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            return client
        except Exception as e:
            logging.warning(f"Session.client('s3') failed: {e}. Falling back to boto3.client.")

    # Final fallback: direct boto3.client
    try:
        client = boto3.client(
            "s3",
            config=config,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        return client
    except Exception as e:
        logging.error(f"Failed to initialize boto3 S3 client: {e}")
        return None


def head_object_safe(s3, bucket: str, key: str) -> Optional[Dict]:
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        # Distinguish not found
        if hasattr(e, "response"):
            code = e.response.get("Error", {}).get("Code", "")
            http = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if code in ("404", "NoSuchKey", "NotFound") or http == 404:
                return None
        # Other errors (permission, transient, etc.) - propagate up
        raise


def guess_content_type(file_path: Path) -> Optional[str]:
    ctype, _ = mimetypes.guess_type(str(file_path))
    return ctype


def needs_upload(
    local_path: Path,
    head: Optional[Dict],
    checksum: bool,
    mtime_tolerance_seconds: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Return (should_upload, md5_if_computed)
    """
    if head is None:
        return True, None
    st = local_path.stat()
    local_size = st.st_size
    remote_size = int(head.get("ContentLength", -1))
    if remote_size != local_size:
        return True, None
    if checksum:
        # Prefer metadata md5 when present
        remote_meta_md5 = (head.get("Metadata") or {}).get("md5")
        local_md5 = compute_md5(local_path)
        if remote_meta_md5:
            if remote_meta_md5 == local_md5:
                return False, local_md5
            return True, local_md5
        # Fallback to ETag when it is a single-part upload (no '-')
        etag = str(head.get("ETag", "")).strip('"')
        if etag and "-" not in etag:
            if etag == local_md5:
                return False, local_md5
            return True, local_md5
        # Multipart ETag is unreliable -> fallback to mtime
    # mtime-based comparison for default path and checksum fallback
    s3_last_modified: datetime = head.get("LastModified")  # tz-aware
    local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
    tolerance = timedelta(seconds=mtime_tolerance_seconds)
    if local_mtime > (s3_last_modified + tolerance):
        return True, None
    return False, None


def needs_download(
    local_path: Path,
    head: Optional[Dict],
    checksum: bool,
    mtime_tolerance_seconds: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Return (should_download, md5_if_computed)
    """
    # If remote object does not exist, nothing to download
    if head is None:
        return False, None
    if not local_path.exists():
        return True, None
    st = local_path.stat()
    local_size = st.st_size
    remote_size = int(head.get("ContentLength", -1))
    if remote_size != local_size:
        return True, None
    if checksum:
        # Prefer metadata md5 when present
        remote_meta_md5 = (head.get("Metadata") or {}).get("md5")
        local_md5 = compute_md5(local_path)
        if remote_meta_md5:
            if remote_meta_md5 == local_md5:
                return False, local_md5
            return True, local_md5
        # Fallback to ETag when it is a single-part upload (no '-')
        etag = str(head.get("ETag", "")).strip('"')
        if etag and "-" not in etag:
            if etag == local_md5:
                return False, local_md5
            return True, local_md5
        # Multipart ETag is unreliable -> fallback to mtime
    # mtime-based comparison for default path and checksum fallback
    s3_last_modified: datetime = head.get("LastModified")  # tz-aware
    local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
    tolerance = timedelta(seconds=mtime_tolerance_seconds)
    if s3_last_modified > (local_mtime + tolerance):
        return True, None
    return False, None


def list_local_files(src: Path, includes: List[str], excludes: List[str], skip_dirs: Set[str]) -> List[Path]:
    files: List[Path] = []
    for p in src.rglob("*"):
        if p.is_file():
            rel_path = p.relative_to(src)
            if path_has_skip_dir(rel_path, skip_dirs):
                continue
            rel = rel_path.as_posix()
            if should_include(rel, includes, excludes):
                files.append(p)
    return files


def delete_remote_objects(s3, bucket: str, keys_to_delete: Set[str]) -> Tuple[int, int]:
    """
    Delete remote objects in batches of 1000. Returns (deleted_count, failed_count)
    """
    if not keys_to_delete:
        return 0, 0
    deleted = 0
    failed = 0
    keys = list(keys_to_delete)
    for i in range(0, len(keys), 1000):
        chunk = keys[i:i + 1000]
        try:
            resp = s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True}
            )
            deleted += len(resp.get("Deleted", []))
            errors = resp.get("Errors", [])
            failed += len(errors)
            for err in errors:
                logging.warning(f"Delete failed: {err}")
        except Exception as e:
            failed += len(chunk)
            logging.error(f"Delete batch failed ({len(chunk)}): {e}")
    return deleted, failed


def paginate_remote_keys(s3, bucket: str, prefix: str) -> Set[str]:
    keys: Set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix
    for page in paginator.paginate(**kwargs):
        contents = page.get("Contents", [])
        for obj in contents:
            keys.add(obj["Key"])
    return keys


def s3_sync_boto3(
    src: Path,
    bucket: str,
    prefix: str,
    region: Optional[str],
    profile: Optional[str],
    includes: List[str],
    excludes: List[str],
    skip_dirs: Set[str],
    delete: bool,
    checksum: bool,
    workers: int,
    dry_run: bool,
    multipart_threshold: int,
    multipart_chunksize: int,
    endpoint_url: Optional[str],
    access_key: Optional[str],
    secret_key: Optional[str],
    public_url_base: Optional[str],
    print_urls: bool,
    acl: Optional[str],
) -> int:
    if boto3 is None:
        logging.error("boto3 is not installed. Install with: pip install boto3")
        return 2

    s3 = create_boto3_client(profile, region, endpoint_url, access_key, secret_key)
    if s3 is None:
        return 3

    # Prepare transfer config
    tconfig = TransferConfig(
        multipart_threshold=max(5 * 1024 * 1024, int(multipart_threshold)),
        multipart_chunksize=max(5 * 1024 * 1024, int(multipart_chunksize)),
        max_concurrency=max(1, int(workers)),
        use_threads=True,
    )

    prefix_norm = normalize_prefix(prefix)
    skip_set = skip_dirs or set()
    files = list_local_files(src, includes, excludes, skip_set)
    stats = Stats(scanned=len(files), dry_run=dry_run)
    base_url = public_url_base.rstrip("/") if public_url_base else None

    # Plan uploads
    planned: List[Tuple[Path, str]] = []
    present_keys: Set[str] = set()
    for p in tqdm(files):
        rel_posix = to_posix_rel(p, src)
        key = f"{prefix_norm}/{rel_posix}" if prefix_norm else rel_posix
        present_keys.add(key)
        # Head check
        head: Optional[Dict] = None
        try:
            head = head_object_safe(s3, bucket, key)
        except Exception as e:
            logging.warning(f"HeadObject failed for s3://{bucket}/{key}: {e}")
        try:
            upload, _md5 = needs_upload(p, head, checksum)
        except Exception as e:
            logging.warning(f"Compare failed for {p}: {e}")
            upload = True  # be conservative

        if upload:
            planned.append((p, key))
        else:
            stats.skipped += 1

    stats.planned = len(planned)

    # Perform uploads
    if dry_run:
        for p, key in planned:
            logging.info(f"[DRY] {p} -> s3://{bucket}/{key}")
            if print_urls and base_url:
                logging.info(f"[DRY][URL] {base_url}/{key}")
    else:
        for idx, (p, key) in enumerate(planned, 1):
            try:
                extra_args: Dict = {}
                ctype = guess_content_type(p)
                if ctype:
                    extra_args["ContentType"] = ctype
                if acl:
                    extra_args["ACL"] = acl
                if checksum:
                    md5_hex = compute_md5(p)
                    extra_args.setdefault("Metadata", {})["md5"] = md5_hex
                s3.upload_file(
                    Filename=str(p),
                    Bucket=bucket,
                    Key=key,
                    ExtraArgs=extra_args if extra_args else None,
                    Config=tconfig,
                )
                stats.uploaded += 1
                if print_urls and base_url:
                    logging.info(f"[URL] {base_url}/{key}")
            except (ClientError, NoCredentialsError) as e:
                logging.error(f"Upload failed [{idx}/{len(planned)}] {p} -> s3://{bucket}/{key}: {e}")
                stats.failed += 1
            except Exception as e:
                logging.error(f"Upload failed [{idx}/{len(planned)}] {p} -> s3://{bucket}/{key}: {e}")
                stats.failed += 1

    # Optional delete
    if delete:
        if not prefix_norm:
            logging.error("--delete is refused without a non-empty --prefix to protect bucket root.")
        else:
            try:
                remote_keys = paginate_remote_keys(s3, bucket, prefix_norm)
                to_delete: Set[str] = set()
                prefix_with_slash = f"{prefix_norm}/"
                for k in remote_keys:
                    if not k.startswith(prefix_with_slash):
                        continue
                    rel = k[len(prefix_with_slash):]
                    if not rel:
                        continue
                    if rel_path_has_skip_dir(rel, skip_set):
                        continue
                    if k not in present_keys:
                        to_delete.add(k)
                if not to_delete:
                    logging.info("No remote objects to delete.")
                else:
                    if dry_run:
                        for k in sorted(to_delete):
                            logging.info(f"[DRY][DELETE] s3://{bucket}/{k}")
                        stats.deleted += len(to_delete)
                    else:
                        deleted, del_failed = delete_remote_objects(s3, bucket, to_delete)
                        stats.deleted += deleted
                        stats.failed += del_failed
            except Exception as e:
                logging.error(f"Delete phase failed: {e}")
                # Do not convert to non-zero exit if uploads were fine

    logging.info(
        f"Done. scanned={stats.scanned}, planned={stats.planned}, "
        f"uploaded={stats.uploaded}, skipped={stats.skipped}, deleted={stats.deleted}, failed={stats.failed}"
    )
    return 0 if stats.failed == 0 else 4


def s3_sync_boto3_download(
    local: Path,
    bucket: str,
    prefix: str,
    region: Optional[str],
    profile: Optional[str],
    includes: List[str],
    excludes: List[str],
    skip_dirs: Set[str],
    delete: bool,
    checksum: bool,
    workers: int,
    dry_run: bool,
    multipart_threshold: int,
    multipart_chunksize: int,
    endpoint_url: Optional[str],
    access_key: Optional[str],
    secret_key: Optional[str],
) -> int:
    if boto3 is None:
        logging.error("boto3 is not installed. Install with: pip install boto3")
        return 2

    s3 = create_boto3_client(profile, region, endpoint_url, access_key, secret_key)
    if s3 is None:
        return 3

    # Prepare transfer config
    tconfig = TransferConfig(
        multipart_threshold=max(5 * 1024 * 1024, int(multipart_threshold)),
        multipart_chunksize=max(5 * 1024 * 1024, int(multipart_chunksize)),
        max_concurrency=max(1, int(workers)),
        use_threads=True,
    )

    prefix_norm = normalize_prefix(prefix)
    skip_set = skip_dirs or set()

    # Gather remote keys with include/exclude filtering
    remote_keys: List[str] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        kwargs = {"Bucket": bucket}
        if prefix_norm:
            kwargs["Prefix"] = prefix_norm
        for page in paginator.paginate(**kwargs):
            contents = page.get("Contents", [])
            for obj in contents:
                key = obj["Key"]
                if prefix_norm:
                    if not key.startswith(prefix_norm):
                        continue
                    # Compute relative path under prefix
                    rel = key[len(prefix_norm) + 1:] if len(key) > len(prefix_norm) else ""
                else:
                    rel = key
                if not rel:
                    continue
                if should_include(rel, includes, excludes):
                    if rel_path_has_skip_dir(rel, skip_set):
                        continue
                    remote_keys.append(key)
    except Exception as e:
        logging.error(f"Failed to list remote objects: {e}")
        return 5

    stats = Stats(scanned=len(remote_keys), dry_run=dry_run)

    # Plan downloads
    planned: List[Tuple[str, Path]] = []
    present_rel_set: Set[str] = set()
    for key in tqdm(remote_keys):
        rel = key[len(prefix_norm) + 1:] if prefix_norm else key
        if not rel:
            continue
        present_rel_set.add(rel)
        local_path = local / rel

        head: Optional[Dict] = None
        try:
            head = head_object_safe(s3, bucket, key)
        except Exception as e:
            logging.warning(f"HeadObject failed for s3://{bucket}/{key}: {e}")

        try:
            if not local_path.exists():
                should_dl = True
            else:
                should_dl, _md5 = needs_download(local_path, head, checksum)
        except Exception as e:
            logging.warning(f"Compare failed for {local_path}: {e}")
            should_dl = True  # be conservative

        if should_dl:
            planned.append((key, local_path))
        else:
            stats.skipped += 1

    stats.planned = len(planned)

    # Perform downloads
    if dry_run:
        for key, lpath in planned:
            logging.info(f"[DRY] s3://{bucket}/{key} -> {lpath}")
    else:
        for idx, (key, lpath) in enumerate(planned, 1):
            try:
                lpath.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=str(lpath),
                    Config=tconfig,
                )
                stats.uploaded += 1
            except (ClientError, NoCredentialsError) as e:
                logging.error(f"Download failed [{idx}/{len(planned)}] s3://{bucket}/{key} -> {lpath}: {e}")
                stats.failed += 1
            except Exception as e:
                logging.error(f"Download failed [{idx}/{len(planned)}] s3://{bucket}/{key} -> {lpath}: {e}")
                stats.failed += 1

    # Optional local delete
    if delete:
        if not prefix_norm:
            logging.error("--delete is refused without a non-empty --prefix to protect local root.")
        else:
            try:
                local_files = list_local_files(local, includes, excludes, skip_set)
                to_delete_paths: List[Path] = []
                for p in local_files:
                    rel = to_posix_rel(p, local)
                    if rel not in present_rel_set:
                        to_delete_paths.append(p)

                if not to_delete_paths:
                    logging.info("No local files to delete.")
                else:
                    if dry_run:
                        for p in sorted(to_delete_paths):
                            logging.info(f"[DRY][DELETE] {p}")
                        stats.deleted += len(to_delete_paths)
                    else:
                        for p in to_delete_paths:
                            try:
                                p.unlink()
                                stats.deleted += 1
                            except Exception as e:
                                logging.error(f"Delete failed for {p}: {e}")
                                stats.failed += 1
            except Exception as e:
                logging.error(f"Local delete phase failed: {e}")
                # Do not convert to non-zero exit if downloads were fine

    logging.info(
        f"Done. scanned={stats.scanned}, planned={stats.planned}, "
        f"downloaded={stats.uploaded}, skipped={stats.skipped}, deleted={stats.deleted}, failed={stats.failed}"
    )
    return 0 if stats.failed == 0 else 4


def s3_sync_cli(
    direction: str,
    local: Path,
    bucket: str,
    prefix: str,
    region: Optional[str],
    profile: Optional[str],
    includes: List[str],
    excludes: List[str],
    delete: bool,
    dry_run: bool,
    checksum: bool,  # ignored but warn
    endpoint_url: Optional[str],
    acl: Optional[str],
    skip_dirs: Iterable[str],
) -> int:
    skip_set = set(skip_dirs or [])
    if not detect_aws_cli():
        logging.warning("aws CLI not found. Falling back to boto3 implementation.")
        if direction == "download":
            return s3_sync_boto3_download(
                local=local,
                bucket=bucket,
                prefix=prefix,
                region=region,
                profile=profile,
                includes=includes,
                excludes=excludes,
                delete=delete,
                checksum=checksum,
                workers=8,
                dry_run=dry_run,
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                endpoint_url=endpoint_url,
                access_key=None,
                secret_key=None,
                skip_dirs=skip_set,
            )
        else:
            return s3_sync_boto3(
                src=local,
                bucket=bucket,
                prefix=prefix,
                region=region,
                profile=profile,
                includes=includes,
                excludes=excludes,
                delete=delete,
                checksum=checksum,
                workers=8,
                dry_run=dry_run,
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                endpoint_url=endpoint_url,
                access_key=None,
                secret_key=None,
                public_url_base=None,
                print_urls=False,
                acl=None,
                skip_dirs=skip_set,
            )
    if checksum:
        logging.warning("--checksum is ignored in aws CLI mode; CLI's own comparison (size+mtime) will be used.")

    cmd = build_aws_cli_sync_command(
        direction=direction,
        local=local,
        bucket=bucket,
        prefix=prefix,
        region=region,
        profile=profile,
        includes=includes,
        excludes=excludes,
        skip_dirs=skip_dirs,
        delete=delete,
        dry_run=dry_run,
        endpoint_url=endpoint_url,
        acl=acl,
    )
    logging.debug(f"Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,  # 自动解码为文本
            encoding='utf-8', # 明确指定编码，避免乱码
            errors='replace', # 如果解码失败，用'?'替换
            bufsize=1  # 行缓冲
        )
        # 实时逐行读取输出
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush() # 确保立即显示
        proc.wait()
        if proc.returncode != 0:
            logging.error(f"命令 '{' '.join(cmd)}' 执行失败，退出码: {proc.returncode}")
        return proc.returncode

    except FileNotFoundError:
        logging.warning("aws CLI not found at execution time. Falling back to boto3 implementation.")
        if direction == "download":
            return s3_sync_boto3_download(
                local=local,
                bucket=bucket,
                prefix=prefix,
                region=region,
                profile=profile,
                includes=includes,
                excludes=excludes,
                delete=delete,
                checksum=checksum,
                workers=8,
                dry_run=dry_run,
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                endpoint_url=endpoint_url,
                access_key=None,
                secret_key=None,
                skip_dirs=skip_set,
            )
        else:
            return s3_sync_boto3(
                src=local,
                bucket=bucket,
                prefix=prefix,
                region=region,
                profile=profile,
                includes=includes,
                excludes=excludes,
                delete=delete,
                checksum=checksum,
                workers=8,
                dry_run=dry_run,
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                endpoint_url=endpoint_url,
                access_key=None,
                secret_key=None,
                public_url_base=None,
                print_urls=False,
                acl=None,
                skip_dirs=skip_set,
            )
    except Exception as e:
        logging.error(f"aws s3 sync failed: {e}")
        return 5


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incrementally sync local ViStory outputs with AWS S3 or compatible endpoints (boto3 or aws CLI)."
    )
    # Direction
    p.add_argument("--direction", choices=["upload", "download"], default="upload",
                   help="Sync direction: 'upload' (local -> S3) or 'download' (S3 -> local). Default: upload")
    # Local path
    p.add_argument("--src", type=str, default="data/outputs_avif",
                   help="Local directory. Acts as source for upload, destination for download. Default: data/outputs")
    p.add_argument("--bucket", type=str, default=os.getenv("AWS_S3_BUCKET", "my-vistory-bucket"), help="Target S3 bucket")
    p.add_argument("--prefix", type=str, default="outputs_avif", help="S3 prefix (no leading slash). Default: outputs")
    p.add_argument(
        "--src-prefix",
        dest="src_prefix_pairs",
        action="append",
        type=parse_src_prefix_pair,
        metavar="SRC=PREFIX",
        help="Define SRC/S3 prefix pairs (format 'local_path=prefix'). Repeat to run multiple syncs. Overrides --src/--prefix when provided.",
    )
    p.add_argument("--region", type=str, default=os.getenv("AWS_REGION", "ap-southeast-1"), help="AWS region (e.g., us-east-005)")
    p.add_argument("--profile", type=str, default=os.getenv("AWS_PROFILE", "default"), help="AWS profile name (default: default)")
    p.add_argument("--endpoint", type=str, default=(os.getenv("AWS_S3_ENDPOINT") or os.getenv("S3_ENDPOINT") or os.getenv("B2_ENDPOINT") or os.getenv("ENDPOINT")), help="S3-compatible endpoint URL (e.g., https://s3.us-east-005.backblazeb2.com)")
    p.add_argument("--acl", type=str, default=os.getenv("AWS_S3_ACL"), help="Canned ACL for uploaded objects (e.g., public-read)")

    # Behavior
    p.add_argument("--use-cli", action="store_true", default=True, help="Use `aws s3 sync` when available; otherwise fallback to boto3")
    p.add_argument("--delete", action="store_true", help="Delete objects not present on the source side (remote for upload, local for download). Requires non-empty prefix")
    p.add_argument("--checksum", action="store_true", help="Use MD5 comparison (boto3 mode). Slower but stricter")
    p.add_argument("--dry-run", action="store_true", help="Print planned actions without performing them")

    # Filters
    p.add_argument("--include", action="append", default=[], help='Include glob (relative POSIX path). Can be repeated')
    p.add_argument("--exclude", action="append", default=[], help='Exclude glob (relative POSIX path). Can be repeated')
    p.add_argument("--skip", action="append", default=['.cache'], help='Skip directories by name (matches any folder segment). Can be repeated')

    # Public URL output (boto3 upload mode only)
    p.add_argument("--public-url-base", type=str, default=(os.getenv("S3_PUBLIC_BASE_URL") or os.getenv("PUBLIC_URL_BASE") or os.getenv("CUSTOM_DOMAIN") or os.getenv("CUSTOM_BASE_URL")), help="Base URL used to print public links for uploaded keys (e.g., https://img.blenet.top)")
    p.add_argument("--print-urls", action="store_true", help="Print public URLs of uploaded files (requires --public-url-base; boto3 upload mode only)")

    # Concurrency and transfer (boto3 mode)
    p.add_argument("--workers", type=int, default=256, help="Concurrent transfers for boto3 mode (default: 8)")
    p.add_argument("--multipart-threshold", type=int, default=8 * 1024 * 1024, help="Multipart threshold in bytes (default: 8MB)")
    p.add_argument("--multipart-chunk-size", type=int, default=8 * 1024 * 1024, help="Multipart chunk size in bytes (default: 8MB)")

    # Logging
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")

    return p.parse_args(argv)


def guess_outputs_root(base: Path) -> Path:
    # Mirror behavior from upload_to_oss: auto-step into 'outputs' when present
    if (base / "outputs").is_dir():
        return base / "outputs"
    return base


def run_sync_for_pair(
    *,
    pair_idx: int,
    pair_total: int,
    args: argparse.Namespace,
    local_path: str,
    prefix_value: str,
    bucket: str,
    region: Optional[str],
    profile: Optional[str],
    endpoint_url: Optional[str],
    skip_dirs: List[str],
    skip_set: Set[str],
    access_key_env: Optional[str],
    secret_key_env: Optional[str],
) -> int:
    pair_tag = f"[Pair {pair_idx}/{pair_total}]"
    prefix_norm = normalize_prefix(prefix_value)
    local_base = Path(local_path)
    if args.direction == "upload":
        local_base = guess_outputs_root(local_base)
        if not local_base.exists() or not local_base.is_dir():
            logging.error(f"{pair_tag} Source directory does not exist: {local_base}")
            return 2
    else:
        if not local_base.exists():
            try:
                local_base.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"{pair_tag} Failed to create destination directory {local_base}: {e}")
                return 2
        elif not local_base.is_dir():
            logging.error(f"{pair_tag} Destination path is not a directory: {local_base}")
            return 2

    s3_display = f"s3://{bucket}/{prefix_norm}" if prefix_norm else f"s3://{bucket}"
    if args.direction == "upload":
        logging.info(f"{pair_tag} Sync plan: {local_base} -> {s3_display}")
    else:
        logging.info(f"{pair_tag} Sync plan: {s3_display} -> {local_base}")

    if args.use_cli:
        rc = s3_sync_cli(
            direction=args.direction,
            local=local_base,
            bucket=bucket,
            prefix=prefix_norm,
            region=region,
            profile=profile,
            includes=args.include or [],
            excludes=args.exclude or [],
            delete=args.delete,
            dry_run=args.dry_run,
            checksum=args.checksum,
            endpoint_url=endpoint_url,
            acl=args.acl,
            skip_dirs=skip_dirs,
        )
    else:
        if args.direction == "upload":
            rc = s3_sync_boto3(
                src=local_base,
                bucket=bucket,
                prefix=prefix_norm,
                region=region,
                profile=profile,
                includes=args.include or [],
                excludes=args.exclude or [],
                delete=args.delete,
                checksum=args.checksum,
                workers=max(1, int(args.workers or 1)),
                dry_run=args.dry_run,
                multipart_threshold=max(5 * 1024 * 1024, int(args.multipart_threshold or (8 * 1024 * 1024))),
                multipart_chunksize=max(5 * 1024 * 1024, int(args.multipart_chunk_size or (8 * 1024 * 1024))),
                endpoint_url=endpoint_url,
                access_key=access_key_env,
                secret_key=secret_key_env,
                public_url_base=args.public_url_base,
                print_urls=args.print_urls,
                acl=args.acl,
                skip_dirs=skip_set,
            )
        else:
            rc = s3_sync_boto3_download(
                local=local_base,
                bucket=bucket,
                prefix=prefix_norm,
                region=region,
                profile=profile,
                includes=args.include or [],
                excludes=args.exclude or [],
                delete=args.delete,
                checksum=args.checksum,
                workers=max(1, int(args.workers or 1)),
                dry_run=args.dry_run,
                multipart_threshold=max(5 * 1024 * 1024, int(args.multipart_threshold or (8 * 1024 * 1024))),
                multipart_chunksize=max(5 * 1024 * 1024, int(args.multipart_chunk_size or (8 * 1024 * 1024))),
                endpoint_url=endpoint_url,
                access_key=access_key_env,
                secret_key=secret_key_env,
                skip_dirs=skip_set,
            )
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    # Load .env with python-dotenv if available
    if load_dotenv:
        try:
            load_dotenv()
            logging.debug("Loaded environment from .env via python-dotenv")
        except Exception as e:
            logging.warning(f"Failed to load .env via python-dotenv: {e}")

    # Overlay args with .env aliases if provided
    bucket_env = get_env_with_aliases(["AWS_S3_BUCKET", "BUCKET_NAME", "B2_BUCKET_NAME", "bucketName"])
    endpoint_env = get_env_with_aliases(["AWS_S3_ENDPOINT", "S3_ENDPOINT", "B2_ENDPOINT", "ENDPOINT", "Endpoint"])
    region_env = get_env_with_aliases(["AWS_REGION", "REGION", "B2_REGION", "地区"])
    access_key_env = get_env_with_aliases(["AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY", "KEY_ID", "keyID"])
    secret_key_env = get_env_with_aliases(["AWS_SECRET_ACCESS_KEY", "AWS_SECRET_KEY", "APPLICATION_KEY", "applicationKey"])
    public_url_env = get_env_with_aliases(["S3_PUBLIC_BASE_URL", "PUBLIC_URL_BASE", "CUSTOM_DOMAIN", "CUSTOM_BASE_URL"])
    src_prefix_env = get_env_with_aliases(["SRC_PREFIX_PAIRS", "SRC_PREFIX_DEFAULTS", "SYNC_SRC_PREFIX_PAIRS"])

    if bucket_env:
        args.bucket = bucket_env
    if endpoint_env and not args.endpoint:
        args.endpoint = endpoint_env
    if region_env:
        args.region = region_env
    if public_url_env and not args.public_url_base:
        args.public_url_base = public_url_env

    # Export to environment for provider chain and aws CLI
    if access_key_env and not os.getenv("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = access_key_env
    if secret_key_env and not os.getenv("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key_env
    if args.region and not os.getenv("AWS_REGION"):
        os.environ["AWS_REGION"] = args.region
    if args.bucket and not os.getenv("AWS_S3_BUCKET"):
        os.environ["AWS_S3_BUCKET"] = args.bucket
    if args.endpoint and not os.getenv("AWS_S3_ENDPOINT"):
        os.environ["AWS_S3_ENDPOINT"] = args.endpoint
    if args.public_url_base and not os.getenv("S3_PUBLIC_BASE_URL"):
        os.environ["S3_PUBLIC_BASE_URL"] = args.public_url_base

    bucket = args.bucket
    if not bucket:
        logging.error("Missing --bucket.")
        return 2

    skip_dirs = normalize_skip_dirs(args.skip or [])
    skip_set = set(skip_dirs)

    env_src_prefix_pairs: List[Tuple[str, str]] = []
    if src_prefix_env:
        try:
            env_src_prefix_pairs = parse_src_prefix_blob(src_prefix_env)
        except ValueError as e:
            logging.error(f"Failed to parse SRC/PREFIX pairs from environment: {e}")
            return 2

    # Resolve profile and region (allow environment to override)
    region = args.region or os.getenv("AWS_REGION")
    profile = args.profile or os.getenv("AWS_PROFILE")
    endpoint_url = args.endpoint

    # If credentials are provided via .env/environment, ignore profile to avoid 'profile not found' errors
    if (os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_SECRET_ACCESS_KEY")):
        if profile:
            logging.info("Environment/.env credentials detected; ignoring AWS profile.")
        profile = None

    pairs: List[Tuple[str, str]]
    if args.src_prefix_pairs:
        pairs = list(args.src_prefix_pairs)
    elif env_src_prefix_pairs:
        pairs = env_src_prefix_pairs
    else:
        pairs = [(args.src, args.prefix)]

    logging.info(f"Preparing {len(pairs)} sync pair(s).")

    logging.info(
        f"mode={'aws-cli' if args.use_cli and detect_aws_cli() else 'boto3'} "
        f"direction={args.direction} region={region} endpoint={endpoint_url or ''} profile={profile} delete={args.delete} checksum={args.checksum} dry_run={args.dry_run}"
    )

    if args.use_cli and args.print_urls:
        logging.warning("--print-urls is only supported in boto3 upload mode; will be ignored.")
    if (not args.use_cli) and args.direction != "upload" and args.print_urls:
        logging.warning("--print-urls is only supported in boto3 upload mode; ignoring for download.")

    overall_rc = 0
    for idx, (src_value, prefix_value) in enumerate(pairs, start=1):
        rc = run_sync_for_pair(
            pair_idx=idx,
            pair_total=len(pairs),
            args=args,
            local_path=src_value,
            prefix_value=prefix_value,
            bucket=bucket,
            region=region,
            profile=profile,
            endpoint_url=endpoint_url,
            skip_dirs=skip_dirs,
            skip_set=skip_set,
            access_key_env=access_key_env,
            secret_key_env=secret_key_env,
        )
        if rc != 0:
            overall_rc = rc
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
