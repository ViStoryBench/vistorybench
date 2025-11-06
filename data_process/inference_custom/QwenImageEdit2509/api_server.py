import os
import io
import base64
import threading
from typing import List, Optional, Tuple, Dict, Any

from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch 
import math
import dotenv
dotenv.load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

import torch
from diffusers import QwenImageEditPlusPipeline

app = FastAPI(title="Qwen Image Edit REST Inference API", version="1.0")

PIPELINE: Optional[QwenImageEditPlusPipeline] = None
DEVICE: Optional[torch.device] = None
pipe_lock = threading.Lock()

NEGATIVE_PROMPT_DEFAULT = os.getenv("NEGATIVE_PROMPT", "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly")

def _env_flag(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default).lower() in ("1", "true", "yes", "y", "on"))

def _get_dtype() -> torch.dtype:
    dtype_str = os.getenv("DTYPE", "bf16").lower()
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32
    return torch.bfloat16

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_pipeline(weight_dtype: torch.dtype) -> Tuple[QwenImageEditPlusPipeline, torch.device]:
    device = _get_device()
    model_path = os.getenv("MODEL_PATH", "")

    # # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
    # scheduler_config = {
    #     "base_image_seq_len": 256,
    #     "base_shift": math.log(3),  # We use shift=3 in distillation
    #     "invert_sigmas": False,
    #     "max_image_seq_len": 8192,
    #     "max_shift": math.log(3),  # We use shift=3 in distillation
    #     "num_train_timesteps": 1000,
    #     "shift": 1.0,
    #     "shift_terminal": None,  # set shift_terminal to None
    #     "stochastic_sampling": False,
    #     "time_shift_type": "exponential",
    #     "use_beta_sigmas": False,
    #     "use_dynamic_shifting": True,
    #     "use_exponential_sigmas": False,
    #     "use_karras_sigmas": False,
    # }
    # scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    # pipe = DiffusionPipeline.from_pretrained(
    #     "/gemini/platform/public/aigc/zhuangcailin/pretrain/Comfy-Org/Qwen-Image-Edit_ComfyUI", scheduler=scheduler, torch_dtype=torch.bfloat16
    # ).to("cuda")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        # scheduler=scheduler,
        torch_dtype=weight_dtype,
    )
    # pipe.load_lora_weights(
    #     "/gemini/platform/public/aigc/zhuangcailin/pretrain/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-2509", weight_name="Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"
    # )
    lora_path = os.getenv("LORA_PATH","").strip()
    if lora_path:
        try:
            pipe.load_lora_weights(lora_path,weight_name='next-scene_lora-v2-3000.safetensors')
            print(f"[INFO] Loaded LoRA weights from: {lora_path}")
        except Exception as e:
            print(f"[WARN] Failed to load LoRA weights: {e}")
    # Offload / device
    if _env_flag("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "false"):
        pipe.enable_sequential_cpu_offload()
    elif _env_flag("ENABLE_MODEL_CPU_OFFLOAD", "false"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    return pipe, device

def _decode_b64_image(data_b64: str) -> Optional[Image.Image]:
    try:
        raw = base64.b64decode(data_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img
    except Exception as e:
        print(f"[WARN] Failed to decode image: {e}")
        return None

def _extract_prompt_and_images(parts: List[Dict[str, Any]]) -> Tuple[str, List[Image.Image]]:
    prompt: Optional[str] = None
    images: List[Image.Image] = []
    saw_last_ref: bool = False
    for part in parts or []:
        txt = part.get("text")
        if isinstance(txt, str):
            if txt.strip().lower() == "last reference shot:":
                saw_last_ref = True
                continue
            if prompt is None:
                prompt = txt
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and isinstance(inline, dict):
            b64 = inline.get("data")
            if isinstance(b64, str):
                img = _decode_b64_image(b64)
                if img is not None:
                    images.append(img)
    if prompt is None:
       prompt = "A photo"
    if images:
        if saw_last_ref:
            prompt += " The first few pictures are reference images of the characters, and the last picture is the result of the previous shot, which is used for reference."
        else:
            prompt += " The pictures are reference images of the characters."
    return prompt, images

@app.on_event("startup")
def _startup():
    global PIPELINE, DEVICE
    dtype = _get_dtype()
    PIPELINE, DEVICE = load_pipeline(dtype)
    print(f"[INFO] Qwen-Image-Edit pipeline loaded on {DEVICE}.")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1beta/models/{api_model}:generateContent")
async def generate_content(api_model: str, request: Request, key: Optional[str] = None):
    global PIPELINE, DEVICE
    if PIPELINE is None or DEVICE is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    contents = body.get("contents") or []
    if not contents:
        raise HTTPException(status_code=400, detail="Missing 'contents'")
    parts = contents[0].get("parts") or []
    prompt, images = _extract_prompt_and_images(parts)
    if not images:
        raise HTTPException(status_code=400, detail="No input images found in 'parts'")
    gen_cfg: Dict[str, Any] = body.get("generationConfig") or {}
    # Dimensions (optional; if None, pipeline computes from input image)
    width = gen_cfg.get("width")
    height = gen_cfg.get("height")
    width = int(width) if width is not None else None
    height = int(height) if height is not None else None
    steps = int(gen_cfg.get("num_inference_steps", os.getenv("DEFAULT_STEPS", "40")))
    true_cfg_scale = float(gen_cfg.get("true_cfg_scale", os.getenv("TRUE_CFG_SCALE", "4.0")))
    guidance_scale = gen_cfg.get("guidance_scale")
    guidance_scale = float(guidance_scale) if guidance_scale is not None else float(os.getenv("GUIDANCE_SCALE", "1.0"))
    n_images = int(gen_cfg.get("num_images_per_prompt", os.getenv("NUM_IMAGES_PER_PROMPT", "1")))
    seed = int(gen_cfg.get("seed", os.getenv("SEED", "0")))
    negative_prompt = str(gen_cfg.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT))
    # Generator
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.manual_seed(seed)
    print(f"[REQ] steps={steps}, true_cfg_scale={true_cfg_scale}, guidance_scale={guidance_scale}, n_images={n_images}, seed={seed}, hw=({width}x{height}), n_refs={len(images)}")
    with pipe_lock:
        print("prompt "+prompt)
        print("negative_prompt "+negative_prompt)
        print("true_cfg_scale "+str(true_cfg_scale))
        results = PIPELINE(
            image=images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=steps,
            # guidance_scale=guidance_scale,
            num_images_per_prompt=n_images,
            generator=generator,
            output_type="pil",
        )
    if not results or not getattr(results, "images", None):
        raise HTTPException(status_code=500, detail="Generation failed")
    img0: Image.Image = results.images[0]
    buf = io.BytesIO()
    img0.save(buf, format="JPEG", quality=95)
    data_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    resp = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mime_type": "image/jpeg", "data": data_b64}}
                    ]
                }
            }
        ]
    }
    return JSONResponse(content=resp)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8800"))
    uvicorn.run("api_server:app", host=host, port=port, workers=1)