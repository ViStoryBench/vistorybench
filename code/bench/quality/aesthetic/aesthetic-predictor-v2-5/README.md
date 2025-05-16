# SigLIP-based Aesthetic Score Predictor

来自于：https://github.com/discus0434/aesthetic-predictor-v2-5

这个模型可能：“has been improved to evaluate a wider range of image domains such as illustrations.”

## 如何使用

```bash
export HF_HOME="/data/midjourney/yugang/lib/huggingface"
```

``` python
import torch
from PIL import Image

model, processor = torch.hub.load(
    "/data/midjourney/ruiwang/code/vgo/tools/hub/aesthetic-predictor-v2-5", 
    'aesthetic_predictor_v2_5', 
    predictor_name_or_path='/data/midjourney/ruiwang/pretrained/aesthetic_predictor_v2_5.pth', 
    source="local"
)
model = model.to(torch.bfloat16).cuda()

pixel_values = (
    processor(
        images=Image.open("/data/midjourney/ruiwang/assets/example_images/16448529.png").convert("RGB"), return_tensors="pt"
    ).pixel_values.to(torch.bfloat16)
    .cuda()
)

# predict aesthetic score
with torch.inference_mode():
    score = model(pixel_values).logits.squeeze().float().cpu().numpy()

```

## 在线体验地址

https://huggingface.co/spaces/discus0434/aesthetic-predictor-v2-5