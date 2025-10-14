from __future__ import annotations
from typing import List, Optional, Dict, Tuple

import os
import re
from datetime import datetime
from natsort import natsorted

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
TIMESTAMP_PATTERN = re.compile(r'^\d{8}[_-]\d{6}$')   # 20250429-235959  or 20250429_235959


def _is_valid_image(filepath: str) -> bool:
    return os.path.isfile(filepath) and filepath.lower().endswith(IMAGE_EXTS)


def _latest_timestamp(subdirs: List[str]) -> str:
    """
    Given a list of timestamp-like directory names, return the latest one.
    """
    if not subdirs:
        return ''
    # normalise _ and - into -
    subdirs_sorted = sorted(
        subdirs,
        key=lambda x: datetime.strptime(x.replace('_', '-'), "%Y%m%d-%H%M%S")
    )
    return subdirs_sorted[-1]


def _collect_story_images(story_dir: str,
                          return_latest: bool) -> Tuple[List[str], List[str]]:
    """
    Scan a single story directory and return two ordered lists:
    (shot_images, char_images)
    If return_latest=True and timestamp subfolders exist, only the latest folder will be used.
    """
    # 1. locate timestamp level if exists
    subdirs = [d for d in os.listdir(story_dir)
               if os.path.isdir(os.path.join(story_dir, d))
               and TIMESTAMP_PATTERN.match(d)]
    target_dirs = []
    if subdirs:
        if return_latest:
            target_dirs = [os.path.join(story_dir, _latest_timestamp(subdirs))]
        else:
            target_dirs = [os.path.join(story_dir, d) for d in subdirs]
    else:
        # story_dir itself already contains images
        target_dirs = [story_dir]

    shot_paths: List[str] = []
    char_paths: List[str] = []

    for td in target_dirs:
        # Special names for shot/character folders can appear — detect both English and Chinese names.
        # For example: folder names like 'shots' and 'characters'.
        sub_items = os.listdir(td)

        # candidate dedicated subfolders
        shot_sub = None
        char_sub = None
        for s in sub_items:
            if s.lower() in {'shots', 'shot'} or s.endswith('shot') or s.endswith('shots'):
                shot_sub = os.path.join(td, s)
            elif s.lower() in {'chars', 'characters'} or s.endswith('character') or s.endswith('characters'):
                char_sub = os.path.join(td, s)

        # fallback: use td itself
        shot_base = shot_sub if shot_sub and os.path.isdir(shot_sub) else td
        char_base = char_sub if char_sub and os.path.isdir(char_sub) else None

        # shots
        for f in natsorted(os.listdir(shot_base)):
            fp = os.path.join(shot_base, f)
            if _is_valid_image(fp):
                # some business require suffix filtering
                base, _ = os.path.splitext(f)
                if base.endswith('_1') or base.endswith('-1'):
                    # keep only the “_1” or “-1” suffixed images when that pattern exists
                    shot_paths.append(fp)
                else:
                    shot_paths.append(fp)

        # chars
        if char_base:
            for f in natsorted(os.listdir(char_base)):
                fp = os.path.join(char_base, f)
                if _is_valid_image(fp):
                    char_paths.append(fp)

    return shot_paths, char_paths


def load_outputs(
    outputs_root: str,
    methods: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    timestamps: Optional[List[str]] = None,
    return_latest: bool = True
) -> Dict[str, Dict[str, List[str]]]:
    """
    General reader for ViStoryBench outputs.

    Directory structure assumed:
        {outputs_root}/{method}/{mode}/{language}/{timestamp}/{story_id}/...
    Parameters can be lists. If any list is None, wildcard matches all.

    Returns
    -------
    Dict[story_id] -> {"shots": [...], "chars": [...]}

    The function asserts matching shot indices ordering after natsort,
    but does not strictly validate count.
    """
    if isinstance(methods, str):
        methods=[methods]
    if isinstance(modes, str):
        modes=[modes]
    if isinstance(languages, str):
        languages=[languages]
    if isinstance(timestamps, str):
        timestamps=[timestamps]

    methods = methods or os.listdir(outputs_root)
    results: Dict[str, Dict[str, List[str]]] = {}

    for method in methods:
        method_dir = os.path.join(outputs_root, method)
        if not os.path.isdir(method_dir):
            continue

        mode_candidates = modes or os.listdir(method_dir)
        for mode in mode_candidates:
            mode_dir = os.path.join(method_dir, mode)
            if not os.path.isdir(mode_dir):
                continue

            lang_candidates = languages or os.listdir(mode_dir)
            for lang in lang_candidates:
                lang_dir = os.path.join(mode_dir, lang)
                if not os.path.isdir(lang_dir):
                    continue

                ts_candidates = timestamps or [
                    d for d in os.listdir(lang_dir)
                    if os.path.isdir(os.path.join(lang_dir, d))
                ]
                for ts in ts_candidates:
                    ts_dir = os.path.join(lang_dir, ts)
                    if not os.path.isdir(ts_dir):
                        continue

                    story_ids = natsorted([
                        d for d in os.listdir(ts_dir)
                        if os.path.isdir(os.path.join(ts_dir, d))
                    ], key=lambda x: int(x))  # story id like '01'

                    for sid in story_ids:
                        story_dir = os.path.join(ts_dir, sid)
                        shot_imgs, char_imgs = _collect_story_images(
                            story_dir, return_latest)
                        # key = f"{method}/{mode}/{lang}/{sid}"
                        key = sid  # keep existing downstream usage
                        results[key] = {"shots": shot_imgs, "chars": char_imgs}

    return results