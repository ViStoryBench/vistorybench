#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViStoryBench - Sora2 批量生成（云雾 API）
功能：
- 每个 story 仅取前 5 个 shot，将 prompt 合并为一个总提示
- 加入角色参考（从前 5 个 shot 的 image_paths 去重收集）
- 调用 http://yunwu.ai/v1/video/create 创建视频任务（sora-2）
- 轮询 http://yunwu.ai/v1/video/query 直到完成，下载并保存为 MP4
- 输出目录层级保持：{outputs}/{METHOD_SAVE}/{mode}/{language}/{timestamp}/{story_id}/video.mp4

使用示例：
  python data_process/inference_custom/Sora2/run_custom.py --language en --timestamp 20251111_120000
  python data_process/inference_custom/Sora2/run_custom.py --language ch --story_ids 01,02 --server_url http://yunwu.ai/v1/video/create --query_url http://yunwu.ai/v1/video/query --timestamp 20251111_120000

环境变量（可选）：
  export SORA2_SERVER_URL="http://yunwu.ai/v1/video/create"
  export SORA2_QUERY_URL="http://yunwu.ai/v1/video/query"
  export SORA2_API_KEY="YOUR_TOKEN"
"""

import os
import sys
import argparse
import time
import base64
import json
import mimetypes
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

import yaml
import requests

# 将仓库根目录加入 sys.path，便于导入 StoryDataset
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from vistorybench.dataset_loader.dataset_load import StoryDataset

DEFAULT_METHOD = "sora-2"
METHOD_SAVE = "Sora2"
DEFAULT_MODE = "ALL_TextOnly"
DEFAULT_LANGUAGE = "en"
ENV_SERVER_URL = "SORA2_SERVER_URL"
ENV_QUERY_URL = "SORA2_QUERY_URL"
ENV_API_KEY = "API_KEY"

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_root_config(config_path: Optional[str]) -> Dict[str, Any]:
    """读取仓库根目录的 config.yaml（若未提供路径则默认 {repo_root}/config.yaml）"""
    if config_path and os.path.isfile(config_path):
        cfg_path = Path(config_path)
    else:
        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def resolve_paths(args, cfg: Dict[str, Any]) -> Dict[str, str]:
    """基于 config.yaml 与 CLI 参数解析 dataset_root 与 outputs_root"""
    repo_root = Path(__file__).resolve().parents[3]
    ds_cfg = (((cfg or {}).get("core") or {}).get("paths") or {}).get("dataset") or "data/dataset"
    out_cfg = (((cfg or {}).get("core") or {}).get("paths") or {}).get("outputs") or "data/outputs"
    dataset_root = args.dataset_root or str((repo_root / ds_cfg).resolve())
    outputs_root = args.outputs_root or str((repo_root / out_cfg).resolve())
    return {"dataset_root": dataset_root, "outputs_root": outputs_root}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _to_data_url(img_path: str) -> Optional[str]:
    """
    将本地图片文件转为 data URL（data:image/...;base64,xxx）
    若文件不存在或读取失败返回 None
    """
    try:
        if not img_path or not os.path.isfile(img_path):
            return None
        mime, _ = mimetypes.guess_type(img_path)
        if not mime:
            # 默认按 JPEG 处理
            mime = "image/jpeg"
        with open(img_path, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        _log(f"[WARN] 读取/编码图片失败: {img_path} | {e}")
        return None

def _extract_task_id(data: Dict[str, Any]) -> Optional[str]:
    """
    尝试从创建任务返回 JSON 中提取任务 id（兼容不同结构）
    """
    if not isinstance(data, dict):
        return None
    # 常见位置：顶层 id
    tid = data.get("id")
    if isinstance(tid, str) and tid:
        return tid
    # 嵌套 detail.id
    detail = data.get("detail") or {}
    if isinstance(detail, dict):
        tid2 = detail.get("id")
        if isinstance(tid2, str) and tid2:
            return tid2
    # result.id
    result = data.get("result") or {}
    if isinstance(result, dict):
        tid3 = result.get("id")
        if isinstance(tid3, str) and tid3:
            return tid3
    return None

def submit_video_task(
    server_url: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
    timeout: float = 120.0,
    retries: int = 3,
    sleep: float = 2.0,
) -> Optional[str]:
    """
    调用创建视频接口，返回任务 id
    """
    url = server_url.rstrip("/")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for attempt in range(1, retries + 1):
        try:
            _log(f"[REQ] POST {url} attempt {attempt}/{retries}")
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if not (200 <= resp.status_code < 300):
                _log(f"[WARN] Non-2xx response: {resp.status_code} {resp.text[:200]}")
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json()
            tid = _extract_task_id(data)
            if not tid:
                _log(f"[ERROR] 未能从响应中提取任务 id: {json.dumps(data)[:400]}")
                return None
            _log(f"[TASK] 创建成功，任务 id: {tid}")
            return tid
        except Exception as e:
            if attempt < retries:
                delay = sleep * (2 ** (attempt - 1))
                _log(f"[INFO] 创建失败: {e} | {delay:.1f}s 后重试")
                time.sleep(delay)
            else:
                _log(f"[ERROR] 多次失败，放弃创建任务: {e}")
                return None

def _extract_video_url(data: Dict[str, Any]) -> Optional[str]:
    """
    从查询结果 JSON 中尽可能提取视频下载地址
    """
    # prefer 顶层 video_url
    if isinstance(data, dict):
        vu = data.get("video_url")
        if isinstance(vu, str) and vu:
            return vu
        # 尝试 detail.video_url
        detail = data.get("detail")
        if isinstance(detail, dict):
            vu2 = detail.get("video_url")
            if isinstance(vu2, str) and vu2:
                return vu2
            # 兼容其他嵌套
            output = detail.get("output") if isinstance(detail, dict) else None
            if isinstance(output, dict):
                for k in ("video_url", "url"):
                    if isinstance(output.get(k), str) and output.get(k):
                        return output.get(k)
    return None

def poll_video_until_ready(
    query_url: str,
    api_key: Optional[str],
    task_id: str,
    poll_interval: float = 5.0,
    timeout: float = 1800.0,
) -> Optional[str]:
    """
    轮询查询任务，直到完成，返回 video_url；失败/超时返回 None
    """
    url = query_url.rstrip("/")
    deadline = time.time() + timeout
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    params = {"id": task_id}
    last_status = ""
    while time.time() < deadline:
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            if not (200 <= resp.status_code < 300):
                _log(f"[WARN] Query {resp.status_code}: {resp.text[:200]}")
            else:
                data = resp.json()
                status = data.get("status") or data.get("state") or ""
                if status and status != last_status:
                    _log(f"[TASK] 状态变更: {status}")
                    last_status = status
                if (status or "").lower() in {"completed", "succeeded", "success", "done"}:
                    vu = _extract_video_url(data)
                    if vu:
                        _log(f"[TASK] 完成，获取到视频地址")
                        return vu
                    else:
                        _log(f"[WARN] 状态已完成但未找到 video_url，原始响应: {json.dumps(data)[:400]}")
                        return None
                if (status or "").lower() in {"failed", "error", "cancelled", "canceled"}:
                    _log(f"[ERROR] 任务失败: {json.dumps(data)[:400]}")
                    return None
        except Exception as e:
            _log(f"[WARN] 查询异常: {e}")
        time.sleep(poll_interval)
    _log("[ERROR] 轮询超时")
    return None

def download_file(url: str, out_path: str) -> bool:
    ensure_dir(os.path.dirname(out_path))
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        _log(f"[ERROR] 下载失败: {url} | {e}")
        return False

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViStoryBench Sora2 批量生成（按 story 前 5 个 shot 合并）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 配置与路径
    p.add_argument("--config", type=str, default=None, help="config.yaml 路径；默认读取仓库根目录 config.yaml")
    p.add_argument("--dataset_root", type=str, default=None, help="覆盖数据集根目录")
    p.add_argument("--outputs_root", type=str, default=None, help="覆盖输出根目录")
    p.add_argument("--split", type=str, choices=["lite", "full"], default="lite", help="选择 story 列表 split")
    # 语言 / 模型 / 模式 / 时间戳
    p.add_argument("--language", type=str, choices=["ch", "en"], default=DEFAULT_LANGUAGE)
    p.add_argument("--method", type=str, default=DEFAULT_METHOD, help="模型名称（云雾）")
    p.add_argument("--mode", type=str, default=DEFAULT_MODE, help="目录分区名（不发送到 API）")
    p.add_argument("--timestamp", type=str, default=None, help="YYYYMMDD_HHMMSS；缺省则取当前时间")
    # story 选择
    p.add_argument("--story_ids", type=str, default=None, help="以逗号分隔的 story IDs，如 01,02")
    # 服务与鉴权
    p.add_argument("--server_url", type=str, default=os.getenv(ENV_SERVER_URL, "http://yunwu.ai/v1/video/create"), help=f"创建视频接口，或设置环境变量 {ENV_SERVER_URL}")
    p.add_argument("--query_url", type=str, default=os.getenv(ENV_QUERY_URL, "http://yunwu.ai/v1/video/query"), help=f"查询任务接口，或设置环境变量 {ENV_QUERY_URL}")
    p.add_argument("--api_key", type=str, default=os.getenv(ENV_API_KEY, ""), help=f"Bearer Token，也可通过环境变量 {ENV_API_KEY} 提供")
    # 生成参数
    p.add_argument("--orientation", type=str, choices=["portrait", "landscape"], default="landscape")
    p.add_argument("--size", type=str, choices=["small", "large"], default="large")
    p.add_argument("--duration", type=int, default=15, help="视频时长（秒），示例支持 10 或 15")
    p.add_argument("--watermark", action="store_true", default=False, help="传递 true/false；默认 false")
    # 查询控制
    p.add_argument("--poll_interval", type=float, default=5.0, help="轮询间隔（秒）")
    p.add_argument("--query_timeout", type=float, default=1800.0, help="查询超时时间（秒）")
    return p.parse_args()

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
    query_url = args.query_url
    api_key = args.api_key

    # 数据加载
    dataset_dir = os.path.join(dataset_root, "ViStory")
    dataset = StoryDataset(dataset_dir)
    all_story_ids = dataset.get_story_name_list(split=args.split)
    if not all_story_ids:
        _log(f"[FATAL] Story 目录未找到: {dataset_dir}")
        sys.exit(1)

    if args.story_ids:
        wanted = [s.strip() for s in args.story_ids.split(",") if s.strip()]
        story_ids = [s for s in wanted if s in all_story_ids]
        missing = sorted(set(wanted) - set(story_ids))
        if missing:
            _log(f"[WARN] 以下 story IDs 不存在，将被跳过: {','.join(missing)}")
    else:
        story_ids = all_story_ids

    stories = dataset.load_stories(story_ids, language)

    _log(f"[INFO] Dataset: {dataset_dir} | Output: {outputs_root}")
    _log(f"[INFO] method/mode/lang/ts: {method}/{mode}/{language}/{timestamp}")
    _log(f"[INFO] Service: create={server_url} | query={query_url} | Model: {method}")

    for sid in story_ids:
        story = stories.get(sid)
        # 容错
        if not story:
            _log(f"[WARN] 空 story 跳过: {sid}")
            continue
        shots_all = dataset.story_prompt_merge(story, mode='all')
        if not shots_all:
            _log(f"[WARN] story 无 shots 跳过: {sid}")
            continue
        # selected = shots_all[:5]
        selected=shots_all

        # 合并 Prompt
        lines = []
        for i, shot in enumerate(selected):
            sp = str(shot.get("prompt", "")).strip()
            lines.append(f"Shot {i+1:02d}: {sp}")
        combined_prompt = (
            "Generate a coherent short video for the following these shots. "
            "Maintain character identity consistency and overall visual style.\n"
            + "\n".join(lines)
        )

        # 收集角色参考（data URLs）
        uniq_paths: List[str] = []
        seen = set()
        for shot in selected:
            for p in (shot.get("image_paths") or []):
                if p and os.path.isfile(p) and p not in seen:
                    uniq_paths.append(p)
                    seen.add(p)
        images_data: List[str] = []
        for pth in uniq_paths:
            du = _to_data_url(pth)
            if du:
                images_data.append(du)

        out_story_dir = os.path.join(outputs_root, METHOD_SAVE, mode, language, timestamp, sid)
        ensure_dir(out_story_dir)
        out_video_path = os.path.join(out_story_dir, "video.mp4")

        if os.path.isfile(out_video_path):
            _log(f"[SKIP] 已存在，跳过: {sid} -> {out_video_path}")
            continue

        payload = {
            "images": [],
            "model": method,
            "orientation": args.orientation,
            "prompt": combined_prompt,
            "size": args.size,
            "duration": int(args.duration),
            "watermark": bool(args.watermark),
        }

        # 创建任务
        task_id = submit_video_task(server_url, api_key, payload)
        if not task_id:
            _log(f"[ERROR] 创建任务失败，跳过: {sid}")
            continue

        # 轮询
        video_url = poll_video_until_ready(query_url, api_key, task_id, args.poll_interval, args.query_timeout)
        if not video_url:
            _log(f"[ERROR] 未获取到视频地址，跳过保存: {sid}")
            continue

        # 下载保存
        if download_file(video_url, out_video_path):
            _log(f"[SAVE] {sid} -> {out_video_path}")
        else:
            _log(f"[ERROR] 保存失败: {sid} -> {out_video_path}")

    _log("All done.")

if __name__ == "__main__":
    main()