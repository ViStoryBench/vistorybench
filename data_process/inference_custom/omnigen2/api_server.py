import os
import io
import base64
import threading
from typing import List, Optional, Tuple, Dict, Any

import dotenv
dotenv.load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

import torch
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

app = FastAPI(title="OmniGen2 REST Inference API", version="1.0")

accelerator: Optional[Accelerator] = None
pipeline: Optional[OmniGen2Pipeline] = None
pipe_lock = threading.Lock()

NEGATIVE_PROMPT_DEFAULT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

def _env_flag(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default).lower() in ("1", "true", "yes", "y", "on"))

def _get_dtype_and_precision() -> Tuple[torch.dtype, str]:
    dtype_str = os.getenv("DTYPE", "bf16").lower()
    if dtype_str == "fp16":
        return torch.float16, "fp16"
    if dtype_str == "fp32":
        return torch.float32, "no"
    return torch.bfloat16, "bf16"

def load_pipeline_obj(accel: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    model_path = os.getenv("MODEL_PATH", "OmniGen2/OmniGen2")
    transformer_path = os.getenv("TRANSFORMER_PATH", None)
    transformer_lora_path = os.getenv("TRANSFORMER_LORA_PATH", None)
    scheduler_name = os.getenv("SCHEDULER", "euler").lower()

    pipe = OmniGen2Pipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )

    if transformer_path:
        pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=weight_dtype,
        )
    else:
        pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    if transformer_lora_path:
        pipe.load_lora_weights(transformer_lora_path)

    # Caching options
    enable_teacache = _env_flag("ENABLE_TEACACHE", "false")
    enable_taylorseer = _env_flag("ENABLE_TAYLORSEER", "false")
    if enable_teacache and enable_taylorseer:
        print("WARNING: ENABLE_TEACACHE and ENABLE_TAYLORSEER are mutually exclusive. Ignoring ENABLE_TEACACHE.")
    if enable_taylorseer:
        pipe.enable_taylorseer = True
    elif enable_teacache:
        rel_l1 = float(os.getenv("TEACACHE_REL_L1_THRESH", "0.05"))
        pipe.transformer.enable_teacache = True
        pipe.transformer.teacache_rel_l1_thresh = rel_l1

    # Scheduler
    if scheduler_name == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )

    # Offload
    if _env_flag("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "false"):
        pipe.enable_sequential_cpu_offload()
    elif _env_flag("ENABLE_MODEL_CPU_OFFLOAD", "false"):
        pipe.enable_model_cpu_offload()
    elif _env_flag("ENABLE_GROUP_OFFLOAD", "false"):
        apply_group_offloading(pipe.transformer, onload_device=accel.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipe.mllm, onload_device=accel.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipe.vae, onload_device=accel.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipe = pipe.to(accel.device)

    return pipe

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
    
    for part in parts or []:
        # text part
        txt = part.get("text")
        if isinstance(txt, str) and (prompt is None)and txt.strip().lower() != "last reference shot:":
            prompt = txt
        # image part
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and isinstance(inline, dict):
            b64 = inline.get("data")
            if isinstance(b64, str):
                img = _decode_b64_image(b64)
                if img is not None:
                    images.append(img)
    
    prompt+="The first few pictures are reference images of the characters, and the last picture is the result of the previous shot, which is used for reference." if {'text':'last reference shot:'} in parts else  "The pictures are reference images of the characters"

    if prompt is None:
        prompt = "A photo"
    return prompt, images

@app.on_event("startup")
def _startup():
    global accelerator, pipeline
    dtype, mixed_precision = _get_dtype_and_precision()
    accelerator = Accelerator(mixed_precision=mixed_precision)
    pipeline_local = load_pipeline_obj(accelerator, dtype)
    pipeline = pipeline_local
    print("[INFO] OmniGen2 pipeline loaded.")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1beta/models/{api_model}:generateContent")
async def generate_content(api_model: str, request: Request, key: Optional[str] = None):
    global pipeline, accelerator
    if pipeline is None or accelerator is None:
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

    gen_cfg: Dict[str, Any] = body.get("generationConfig") or {}
    width = int(gen_cfg.get("width", os.getenv("DEFAULT_WIDTH", "1344")))
    height = int(gen_cfg.get("height", os.getenv("DEFAULT_HEIGHT", "768")))
    steps = int(gen_cfg.get("num_inference_steps", os.getenv("DEFAULT_STEPS", "50")))
    text_scale = float(gen_cfg.get("text_guidance_scale", os.getenv("TEXT_GUIDANCE_SCALE", "5.0")))
    image_scale = float(gen_cfg.get("image_guidance_scale", os.getenv("IMAGE_GUIDANCE_SCALE", "2.0")))
    cfg_start = float(gen_cfg.get("cfg_range_start", os.getenv("CFG_RANGE_START", "0.0")))
    cfg_end = float(gen_cfg.get("cfg_range_end", os.getenv("CFG_RANGE_END", "1.0")))
    n_images = int(gen_cfg.get("num_images_per_prompt", os.getenv("NUM_IMAGES_PER_PROMPT", "1")))
    seed = int(gen_cfg.get("seed", os.getenv("SEED", "0")))
    scheduler_name = str(gen_cfg.get("scheduler", os.getenv("SCHEDULER", "euler"))).lower()
    negative_prompt = str(gen_cfg.get("negative_prompt", os.getenv("NEGATIVE_PROMPT", NEGATIVE_PROMPT_DEFAULT)))

    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    print(width, height, steps, text_scale, image_scale, cfg_start, cfg_end, n_images, seed, scheduler_name,len(images))
    # Per-request scheduler override
    if scheduler_name == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        pipeline.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
    with pipe_lock:
        results = pipeline(
            prompt=prompt,
            align_res=False,
            input_images=images if images else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            max_sequence_length=1024,
            text_guidance_scale=text_scale,
            image_guidance_scale=image_scale,
            cfg_range=(cfg_start, cfg_end),
            negative_prompt=negative_prompt,
            num_images_per_prompt=n_images,
            generator=generator,
            output_type="pil",
        )

    # Return the first image as JPEG base64
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