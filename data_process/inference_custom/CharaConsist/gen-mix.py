import argparse
import os
import json
import copy
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image

# Pipeline / attention utilities (do NOT modify these files per requirements)
from models.attention_processor_characonsist import (
    reset_attn_processor,
    set_text_len,
    reset_size,
    reset_id_bank,
)
from models.pipeline_characonsist import CharaConsistPipeline


# ----------------------------
# CLI
# ----------------------------
def build_argparser():
    parser = argparse.ArgumentParser(
        description="Generate story shots from ViStoryBench using CharaConsist pipeline with 3-segment prompts."
    )
    # Required
    # parser.add_argument("--story-id", type=str, required=True)

    # Optional - output and control
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--lang", type=str, choices=["en", "ch"], default="en")
    parser.add_argument("--max-shots", type=int, default=0)  # 0 or less = unlimited
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--output-root", type=str, default="CharaConsist/Base")
    parser.add_argument("--primary-character", type=str, default="")
    parser.add_argument("--overwrite", action="store_true", default=False)

    # Pipeline aligned options
    parser.add_argument("--init_mode", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--use_interpolate", action="store_true", default=False)
    parser.add_argument("--share_bg", action="store_true", default=False)
    parser.add_argument("--save_mask", action="store_true", default=False)
    return parser


# ----------------------------
# Model init (copy style from inference.py)
# ----------------------------
def init_model_mode_0(model_path: str):
    pipe = CharaConsistPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda:0")
    return pipe


def init_model_mode_1(model_path: str):
    pipe = CharaConsistPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def init_model_mode_2(model_path: str):
    from diffusers import FluxTransformer2DModel
    from transformers import T5EncoderModel
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="balanced")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, device_map="balanced")
    pipe = CharaConsistPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
        device_map="balanced")
    return pipe


def init_model_mode_3(model_path: str):
    pipe = CharaConsistPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    return pipe


MODEL_INIT_FUNCS = {
    0: init_model_mode_0,
    1: init_model_mode_1,
    2: init_model_mode_2,
    3: init_model_mode_3,
}


# ----------------------------
# Prompt segmentation helpers (reuse logic)
# ----------------------------
def get_text_tokens_length(pipe, p: str) -> int:
    text_mask = pipe.tokenizer_2(
        p,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    ).attention_mask
    return int(text_mask.sum().item() - 1)


def modify_prompt_and_get_length(bg: str, fg: str, act: str, pipe) -> Tuple[str, int, int]:
    # Reuse the exact 3-segment concatenation and length counting
    bg = (bg or "") + " "
    fg = (fg or "") + " "
    prompt = bg + fg + (act or "")
    return prompt, get_text_tokens_length(pipe, bg), get_text_tokens_length(pipe, prompt)


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int], output_path: str):
    # Copied style from inference.py for visualization saving
    img_array = np.array(image).astype(np.float32) * 0.5
    mask_zero = np.zeros_like(img_array)

    mask_resized = Image.fromarray(mask.astype(np.uint8))
    mask_resized = mask_resized.resize(image.size, Image.NEAREST)
    mask_resized = np.array(mask_resized)
    mask_resized = mask_resized[:, :, None]
    color = np.array(color, dtype=np.float32).reshape(1, 1, -1)
    mask_resized_color = mask_resized * color
    img_array = img_array + mask_resized_color * 0.5
    mask_zero = mask_zero + mask_resized_color
    out_img = np.concatenate([img_array, mask_zero], axis=1)
    out_img[out_img > 255] = 255
    out_img = out_img.astype(np.uint8)
    Image.fromarray(out_img).save(output_path)


# ----------------------------
# ViStoryBench JSON parsing (robust)
# ----------------------------
def ci_get(d: Dict, key: str, default=None):
    """Case-insensitive dict getter."""
    for k in d.keys():
        if k.lower() == key.lower():
            return d[k]
    return default


def find_story_dir(story_id: str) -> Optional[str]:
    base = os.path.join("ViStoryBench", "ViStoryBench")
    candidates = []

    # as-is
    candidates.append(os.path.join(base, story_id))
    # remove leading zeros
    try:
        num = int(story_id)
        candidates.append(os.path.join(base, str(num)))
    except Exception:
        pass
    # pad to 2 digits
    try:
        num = int(story_id)
        candidates.append(os.path.join(base, f"{num:02d}"))
    except Exception:
        pass

    for c in candidates:
        story_json = os.path.join(c, "story.json")
        if os.path.isdir(c) and os.path.isfile(story_json):
            return c
    return None


def load_story_json(story_dir: str) -> Dict:
    story_path = os.path.join(story_dir, "story.json")
    with open(story_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_shots_list(story_json: Dict) -> List[Dict]:
    # Prefer Shots → shots → scenes → frames
    for key in ["Shots", "shots", "scenes", "frames"]:
        shots = ci_get(story_json, key, None)
        if isinstance(shots, list) and len(shots) > 0:
            return shots
    return []


def get_lang_fallback_pair(lang: str) -> Tuple[str, str]:
    # primary, secondary
    if lang == "en":
        return "en", "ch"
    return "ch", "en"


def extract_shot_segments(shot: Dict, lang: str) -> Tuple[str, str, str, List[str]]:
    primary, secondary = get_lang_fallback_pair(lang)

    # Background segment
    bg_obj = ci_get(shot, "Setting Description", {}) or {}
    plot_obj = ci_get(shot, "Plot Correspondence", {}) or {}
    bg = (
        (bg_obj.get(primary) or bg_obj.get(secondary)) or
        (plot_obj.get(primary) or "")
    ) or ""

    # Action segment
    act_obj = ci_get(shot, "Static Shot Description", {}) or {}
    act = (
        (act_obj.get(primary) or act_obj.get(secondary)) or
        (plot_obj.get(primary) or "")
    ) or ""

    # Perspective append (optional, compact form)
    persp_obj = ci_get(shot, "Shot Perspective Design", {}) or {}
    persp = (persp_obj.get(primary) or persp_obj.get(secondary) or "").strip()
    if persp:
        # append as " perspective: xxx"
        act = (act or "").rstrip() + f" perspective: {persp}"

    # Characters list
    chars_obj = ci_get(shot, "Characters Appearing", {}) or {}
    chars = (chars_obj.get(primary) or chars_obj.get(secondary) or []) or []
    return bg or "", act or "", persp, chars


def build_character_maps(story_json: Dict) -> Dict[str, Dict]:
    """Return maps for character name and prompts."""
    chars_root = ci_get(story_json, "Characters", {}) or {}
    # Keys in JSON are typically English names, but contain prompt_en/prompt_ch and name_en/name_ch
    name_en_to_detail = {}
    name_ch_to_en = {}
    for _, detail in chars_root.items():
        if not isinstance(detail, dict):
            continue
        name_en = detail.get("name_en") or _
        name_ch = detail.get("name_ch") or ""
        name_en_to_detail[str(name_en)] = detail
        if name_ch:
            name_ch_to_en[str(name_ch)] = str(name_en)
    return {
        "en_to_detail": name_en_to_detail,
        "ch_to_en": name_ch_to_en,
    }


def normalize_to_en(name: str, maps: Dict[str, Dict]) -> str:
    """Try to normalize a provided character name (EN or CH) to EN key."""
    if name in maps["en_to_detail"]:
        return name
    if name in maps["ch_to_en"]:
        return maps["ch_to_en"][name]
    # try raw pass-through
    return name


def get_primary_character(args, shots: List[Dict], maps: Dict[str, Dict]) -> Optional[str]:
    if args.primary_character:
        return normalize_to_en(args.primary_character, maps)

    primary, secondary = get_lang_fallback_pair(args.lang)
    freq: Dict[str, int] = {}
    for shot in shots:
        chars_obj = ci_get(shot, "Characters Appearing", {}) or {}
        names = (chars_obj.get(primary) or chars_obj.get(secondary) or []) or []
        for n in names:
            en_name = normalize_to_en(n, maps)
            freq[en_name] = freq.get(en_name, 0) + 1

    if not freq:
        # fallback: choose first from story-level characters
        if maps["en_to_detail"]:
            return list(maps["en_to_detail"].keys())[0]
        return None

    # pick the highest frequency
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def get_character_prompt(en_name: str, maps: Dict[str, Dict], lang: str) -> str:
    detail = maps["en_to_detail"].get(en_name, {})
    if lang == "en":
        return detail.get("prompt_en", "") or detail.get("prompt_ch", "") or ""
    else:
        return detail.get("prompt_ch", "") or detail.get("prompt_en", "") or ""


def find_ref_image(story_dir: str, en_name: str) -> Optional[str]:
    """Pick reference image: prefer 00.jpg, else lexicographic first of jpg/jpeg/png."""
    base = os.path.join(story_dir, "image", en_name)
    if not os.path.isdir(base):
        # warning only
        return None
    preferred = os.path.join(base, "00.jpg")
    if os.path.isfile(preferred):
        return preferred
    # search
    try:
        files = sorted(os.listdir(base))
    except Exception:
        return None
    for fn in files:
        lower = fn.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
            return os.path.join(base, fn)
    return None


# ----------------------------
# Output helpers
# ----------------------------
def ensure_dirs(root_dir: str, save_mask: bool):
    shots_dir = os.path.join(root_dir, "shots")
    os.makedirs(shots_dir, exist_ok=True)
    mask_dir = None
    if save_mask:
        mask_dir = os.path.join(root_dir, "mask")
        os.makedirs(mask_dir, exist_ok=True)
    return shots_dir, mask_dir


def get_timestamp_str(ts_arg: str) -> str:
    if ts_arg:
        return ts_arg
    # local time
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_header_if_needed(log_path: str):
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("time\tstory_id\tshot_index\toutput_file\tis_id\tused_characters\tref_images\tprompt_preview\tbg_len\treal_len\tuse_interpolate\tshare_bg\tstatus\terror_msg\n")


def append_log(
    log_path: str,
    story_id: str,
    seq_index: int,
    output_file: str,
    is_id: bool,
    used_characters: List[str],
    ref_images: List[str],
    prompt_preview: str,
    bg_len: int,
    real_len: int,
    use_interpolate: bool,
    share_bg: bool,
    status: str,
    error_msg: str = "",
):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{now}\t{story_id}\t{seq_index:02d}\t{output_file}\t{int(is_id)}\t"
            f"{json.dumps(used_characters, ensure_ascii=False)}\t"
            f"{json.dumps(ref_images, ensure_ascii=False)}\t"
            f"{prompt_preview}\t{bg_len}\t{real_len}\t"
            f"{int(use_interpolate)}\t{int(share_bg)}\t{status}\t{error_msg}\n"
        )


def write_plan(plan_path: str, plan_items: List[Dict]):
    with open(plan_path, "w", encoding="utf-8") as f:
        for item in plan_items:
            f.write(f"shot_{item['seq_index']:02d}:\n")
            f.write(f"  BG: {item['bg']}\n")
            f.write(f"  FG: {item['fg']}\n")
            f.write(f"  ACT: {item['act']}\n")
            f.write(f"  Characters: {json.dumps(item['characters'], ensure_ascii=False)}\n")
            f.write(f"  RefImages: {json.dumps(item['ref_images'], ensure_ascii=False)}\n")
            f.write(f"  Output: {item['output_file']}\n")
            f.write("\n")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = build_argparser()
    args = parser.parse_args()

    # GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    # Model path fallback
    model_path = args.model_path if args.model_path else "/path/to/FLUX.1-dev"
    ts = get_timestamp_str(args.timestamp)

    # Resolve story directory
    for story_id in range(1,81):
        story_id=f'{story_id:02d}'
        story_dir = find_story_dir(story_id)
        if story_dir is None:
            raise FileNotFoundError(f"Story directory not found for story_id={story_id}")

        # Load JSON
        story_json = load_story_json(story_dir)
        shots = get_shots_list(story_json)
        if not shots:
            raise RuntimeError("No shots found in story.json (tried Shots/shots/scenes/frames)")

        # Character maps and primary character
        maps = build_character_maps(story_json)
        primary_char_en = get_primary_character(args, shots, maps)
        # Build FG prompt for primary character now (reused for all shots)
        fg_prompt = get_character_prompt(primary_char_en or "", maps, args.lang)

        # Build per-shot segments & character list
        # Also select ID shot: first shot that contains primary character, else first shot
        primary, secondary = get_lang_fallback_pair(args.lang)
        id_shot_pos = 0
        shot_payloads: List[Dict] = []  # per original shot order
        for idx, shot in enumerate(shots):
            bg, act, _, chars = extract_shot_segments(shot, args.lang)
            shot_payloads.append(dict(
                bg=bg,
                act=act,
                characters=chars,
                original_index=idx,
            ))
            if id_shot_pos == 0:  # not set yet
                # check if this shot contains primary character
                if primary_char_en:
                    # shot chars may be EN or CH
                    en_chars = [normalize_to_en(n, maps) for n in chars]
                    if primary_char_en in en_chars:
                        id_shot_pos = idx + 1  # 1-based to distinguish unset from 0
        if id_shot_pos == 0:
            id_shot_pos = 1  # default to first shot

        # Reorder so that ID shot is first, then others in original order
        reordered: List[Dict] = []
        id_item = shot_payloads[id_shot_pos - 1]
        reordered.append(id_item)
        for i, item in enumerate(shot_payloads):
            if i != (id_shot_pos - 1):
                reordered.append(item)

        # Apply max_shots (0 or less means unlimited)
        if args.max_shots and args.max_shots > 0:
            reordered = reordered[:args.max_shots]

        # Prepare output directories and plan
        root_dir = os.path.join(args.output_root, args.lang, f"{ts}/{story_id}")
        shots_dir, mask_dir = ensure_dirs(root_dir, args.save_mask)
        log_path = os.path.join(root_dir, "shots.log")
        log_header_if_needed(log_path)
        plan_path = os.path.join(root_dir, "shot_plan.txt")

        # Collect ref images for logging only
        def collect_ref_images(char_list: List[str]) -> List[str]:
            paths = []
            for name in char_list:
                en_name = normalize_to_en(name, maps)
                rp = find_ref_image(story_dir, en_name)
                paths.append(rp if rp else "")
            return paths

        # Model init (once; we still compute lengths using tokenizer_2)
        pipe = MODEL_INIT_FUNCS[args.init_mode](model_path)
        reset_attn_processor(pipe, size=(args.height // 16, args.width // 16))
        reset_size(pipe, args.height, args.width)

        # Prepare prompts & lengths & plan items
        prompts: List[str] = []
        bg_lens: List[int] = []
        real_lens: List[int] = []
        plan_items: List[Dict] = []
        # FG prompt is global for primary character (consistency)
        for seq_idx, item in enumerate(reordered, start=1):
            bg = item["bg"]
            act = item["act"]
            fg = fg_prompt

            # Skip if all three segments empty
            if not (bg.strip() or fg.strip() or act.strip()):
                # still record plan and log skip
                out_file = os.path.join(shots_dir, f"shot_{seq_idx:02d}.png")
                plan_items.append(dict(
                    seq_index=seq_idx, bg=bg, fg=fg, act=act,
                    characters=item["characters"], ref_images=collect_ref_images(item["characters"]),
                    output_file=out_file,
                ))
                append_log(
                    log_path, story_id, seq_idx, out_file, is_id=(seq_idx == 1),
                    used_characters=item["characters"],
                    ref_images=collect_ref_images(item["characters"]),
                    prompt_preview="",
                    bg_len=0, real_len=0,
                    use_interpolate=args.use_interpolate, share_bg=args.share_bg,
                    status="skip empty segments", error_msg="")
                continue

            prompt, bg_len, real_len = modify_prompt_and_get_length(copy.deepcopy(bg), copy.deepcopy(fg), copy.deepcopy(act), pipe)
            prompts.append(prompt)
            bg_lens.append(bg_len)
            real_lens.append(real_len)

            out_file = os.path.join(shots_dir, f"shot_{seq_idx:02d}.png")
            plan_items.append(dict(
                seq_index=seq_idx, bg=bg, fg=fg, act=act,
                characters=item["characters"], ref_images=collect_ref_images(item["characters"]),
                output_file=out_file,
            ))

        # Dry-run: only output plan file, no model invocation
        write_plan(plan_path, plan_items)
        if args.dry_run:
            return

        # Generation kwargs passed to pipeline
        pipe_kwargs = dict(
            height=args.height,
            width=args.width,
            use_interpolate=args.use_interpolate,
            share_bg=args.share_bg,
        )

        # ID image generation
        if len(prompts) == 0:
            # nothing to generate
            return

        id_prompt = prompts[0]
        frm_prompts = prompts[1:]
        id_bg_len = bg_lens[0]
        id_real_len = real_lens[0]
        id_out_file = os.path.join(shots_dir, "shot_01.png")

        try:
            # We must compute ID spatial state even if output exists (for consistency of following frames)
            set_text_len(pipe, id_bg_len, id_real_len)
            id_images, id_spatial_kwargs = pipe(
                id_prompt, is_id=True, generator=torch.Generator("cpu").manual_seed(args.seed), **pipe_kwargs)

            if not os.path.exists(id_out_file) or args.overwrite:
                id_images[0].save(id_out_file)
                status = "ok"
            else:
                status = "skip existed"

            # Mask saving (optional)
            if args.save_mask and mask_dir:
                try:
                    id_mask_out = os.path.join(mask_dir, "shot_01_mask.jpg")
                    # Save visualization even if image skipped; it's useful for inspection
                    overlay_mask_on_image(id_images[0], id_spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), (255, 0, 0), id_mask_out)
                except Exception as e:
                    # mask save failure should not block generation
                    append_log(
                        log_path, story_id, 1, id_out_file, True,
                        used_characters=reordered[0]["characters"],
                        ref_images=collect_ref_images(reordered[0]["characters"]),
                        prompt_preview=id_prompt,
                        bg_len=id_bg_len, real_len=id_real_len,
                        use_interpolate=args.use_interpolate, share_bg=args.share_bg,
                        status="mask save error", error_msg=str(e))

            append_log(
                log_path, story_id, 1, id_out_file, True,
                used_characters=reordered[0]["characters"],
                ref_images=collect_ref_images(reordered[0]["characters"]),
                prompt_preview=id_prompt,
                bg_len=id_bg_len, real_len=id_real_len,
                use_interpolate=args.use_interpolate, share_bg=args.share_bg,
                status=status, error_msg="")
        except Exception as e:
            append_log(
                log_path, story_id, 1, id_out_file, True,
                used_characters=reordered[0]["characters"],
                ref_images=collect_ref_images(reordered[0]["characters"]),
                prompt_preview=id_prompt,
                bg_len=id_bg_len, real_len=id_real_len,
                use_interpolate=args.use_interpolate, share_bg=args.share_bg,
                status="error", error_msg=str(e))
            # Even if ID fails, we stop subsequent frames as spatial state is missing
            reset_id_bank(pipe)
            return

        # Subsequent frames generation
        spatial_kwargs = dict(id_fg_mask=id_spatial_kwargs["curr_fg_mask"], id_bg_mask=~id_spatial_kwargs["curr_fg_mask"])
        for ind, prompt in enumerate(frm_prompts, start=2):
            out_path = os.path.join(shots_dir, f"shot_{ind:02d}.png")
            bg_len = bg_lens[ind - 1]
            real_len = real_lens[ind - 1]
            chars = reordered[ind - 1]["characters"]
            refs = collect_ref_images(chars)

            # skip existed (idempotence)
            if os.path.exists(out_path) and not args.overwrite:
                append_log(
                    log_path, story_id, ind, out_path, False,
                    used_characters=chars,
                    ref_images=refs,
                    prompt_preview=prompt,
                    bg_len=bg_len, real_len=real_len,
                    use_interpolate=args.use_interpolate, share_bg=args.share_bg,
                    status="skip existed", error_msg="")
                # do NOT run pre-run or final to avoid extra compute
                continue

            try:
                set_text_len(pipe, bg_len, real_len)
                # pre-run to update spatial state
                _, spatial_kwargs = pipe(
                    prompt, is_pre_run=True, generator=torch.Generator("cpu").manual_seed(args.seed),
                    spatial_kwargs=spatial_kwargs, **pipe_kwargs)
                # final generation
                images, spatial_kwargs = pipe(
                    prompt, generator=torch.Generator("cpu").manual_seed(args.seed),
                    spatial_kwargs=spatial_kwargs, **pipe_kwargs)
                images[0].save(out_path)

                # optional mask saving
                if args.save_mask and mask_dir:
                    try:
                        mask_out = os.path.join(mask_dir, f"shot_{ind:02d}_mask.jpg")
                        overlay_mask_on_image(images[0], spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), (255, 0, 0), mask_out)
                    except Exception as e:
                        append_log(
                            log_path, story_id, ind, out_path, False,
                            used_characters=chars,
                            ref_images=refs,
                            prompt_preview=prompt,
                            bg_len=bg_len, real_len=real_len,
                            use_interpolate=args.use_interpolate, share_bg=args.share_bg,
                            status="mask save error", error_msg=str(e))

                append_log(
                    log_path, story_id, ind, out_path, False,
                    used_characters=chars,
                    ref_images=refs,
                    prompt_preview=prompt,
                    bg_len=bg_len, real_len=real_len,
                    use_interpolate=args.use_interpolate, share_bg=args.share_bg,
                    status="ok", error_msg="")
            except Exception as e:
                append_log(
                    log_path, story_id, ind, out_path, False,
                    used_characters=chars,
                    ref_images=refs,
                    prompt_preview=prompt,
                    bg_len=bg_len, real_len=real_len,
                    use_interpolate=args.use_interpolate, share_bg=args.share_bg,
                    status="error", error_msg=str(e))
                # continue to next frame

        reset_id_bank(pipe)


if __name__ == "__main__":
    main()
