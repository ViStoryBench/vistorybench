# from csd_model import CSD_CLIP
# try:
#     import torchvision.transforms.v2 as T
# except ImportError:
#     import torchvision.transforms as T
# from PIL import Image
# import numpy as np
# import torch

# def load_csd():
#     csd_image_encoder = CSD_CLIP(only_global_token=True)
#     state_dict = torch.load('/vepfs-d-data/q-midjourney/chengwei/Projects/ComfyUI/models/clip_vision/csd_vit-large.pth', map_location="cpu")
#     csd_image_encoder.load_state_dict(state_dict, strict=False).to(device = 'cuda')

#     csd_image_encoder = csd_image_encoder.to(dtype=torch.float32)
#     for name, module in csd_image_encoder.named_modules():
#         if isinstance(module, torch.nn.LayerNorm):
#             module = module.to(dtype=torch.float32)
#     return csd_image_encoder

# def encode(image, csd_encoder):
#     sref_preprocess = T.Compose([
#             T.Resize(size=(224, 224), interpolation=T.InterpolationMode.BICUBIC),
#             T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#         ])
#     input_image = sref_preprocess(image).to(device=image.device, dtype=image.dtype)
#     image_embeds = csd_encoder(input_image)['style']
#     return image_embeds

# image = np.array(Image.open('toc-aigc-prod-lipu-feitian-cref-v4.jpg')).astype(np.float32) / 255.
# csd = load_csd()
# image = torch.from_numpy(image).cuda()[None,...].permute(0,3,1,2)
# embeds = encode(image, csd)
# print(embeds)








from .csd_model import CSD_CLIP
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def load_csd():
    csd_image_encoder = CSD_CLIP(only_global_token=True)
    state_dict = torch.load('/data/pretrain/csd/csd_vit-large.pth', map_location="cpu")
    csd_image_encoder.load_state_dict(state_dict, strict=False)
    
    # 冻结所有参数
    csd_image_encoder.eval()
    for param in csd_image_encoder.parameters():
        param.requires_grad = False
        
    # 确保使用float32类型
    csd_image_encoder = csd_image_encoder.to(dtype=torch.float32)
    for module in csd_image_encoder.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.to(dtype=torch.float32)
    
    return csd_image_encoder

def encode(image_tensor,csd_encoder):
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    input_image_tensor = preprocess(image_tensor).to(device=image_tensor.device, dtype=image_tensor.dtype)
    image_embeds = csd_encoder(input_image_tensor)['style']
    return image_embeds

def preprocess(image: Image.Image) -> torch.Tensor:
    """返回带梯度的预处理张量"""
    image_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_array).cuda()[None,...].permute(0,3,1,2)
    return tensor.requires_grad_(True)  # 重要：启用梯度


def force_resize_to_224(image_tensor):
    """暴力缩放任何输入到224x224"""
    return T.functional.resize(
        image_tensor,  # 自动转Tensor并加batch维度
        size=[224, 224], 
        interpolation=T.InterpolationMode.BILINEAR
    )

def get_csd_loss(style_img = None, render_img = None):
    # 初始化模型
    csd_encoder = load_csd().cuda()

    # # 加载并预处理图像
    # if style_img is None:
    #     style_img = Image.open('style_image.jpg').convert('RGB')
    # if render_img is None:
    #     render_img = Image.open('render_image.jpg').convert('RGB')

    # 风格图不需要梯度，渲染图需要梯度
    if isinstance(style_img, (Image.Image)):
        print(f'输入csd的风格图为图像，接下来将转为张量')
        style_tensor = preprocess(style_img).detach()  # detach阻断梯度
    elif isinstance(style_img, (torch.Tensor)):
        style_tensor = style_img

    if isinstance(render_img, (Image.Image)):
        print(f'输入csd的渲染图为图像，接下来将转为张量')
        render_tensor = preprocess(render_img)         # 保持梯度
    elif isinstance(render_img, (torch.Tensor)):
        render_tensor = render_img

    style_tensor = force_resize_to_224(style_tensor) 

    print(f'style_tensor:{style_tensor.shape}')
    print(f'render_tensor:{render_tensor.shape}')

    # 前向计算
    style_embed = encode(style_tensor, csd_encoder)
    render_embed = encode(render_tensor, csd_encoder)

    print(f'style_embed:{style_embed.shape}')
    print(f'render_embed:{render_embed.shape}')

    # 计算余弦相似度损失
    cos_sim = F.cosine_similarity(style_embed, render_embed, dim=-1)
    loss = 1 - cos_sim.mean()

    # 反向传播测试
    # loss.backward()

    print(f'初始损失值: {loss.item():.4f}')
    print(f'渲染图梯度是否存在: {render_tensor.grad is not None}')
    if render_tensor.grad is not None:
        print(f'梯度范数: {render_tensor.grad.norm().item():.4f}')

    return loss



def get_csd_score(csd_encoder = None, img1_path = None, img2_path = None):
    # 初始化模型
    if csd_encoder is None:
        csd_encoder = load_csd().cuda()

    # # 加载并预处理图像
    # if img1 is None:
    #     img1 = Image.open('style_image.jpg').convert('RGB')
    # if img2 is None:
    #     img2 = Image.open('render_image.jpg').convert('RGB')

    Img1 = Image.open(img1_path).convert('RGB')
    image1_tensor = preprocess(Img1).detach() # detach阻断梯度

    Img2 = Image.open(img2_path).convert('RGB')
    image2_tensor = preprocess(Img2).detach() # 保持梯度

    image1_tensor = force_resize_to_224(image1_tensor) 
    image2_tensor = force_resize_to_224(image2_tensor) 

    print(f'image1_tensor:{image1_tensor.shape}')
    print(f'image2_tensor:{image2_tensor.shape}')

    # 前向计算
    image1_embed = encode(image1_tensor, csd_encoder)
    image2_embed = encode(image2_tensor, csd_encoder)

    print(f'image1_embed:{image1_embed.shape}')
    print(f'image2_embed:{image2_embed.shape}')

    # 计算余弦相似度损失
    cos_sim = F.cosine_similarity(image1_embed, image2_embed, dim=-1)
    cos_sim_mean = cos_sim.mean()

    return cos_sim, cos_sim_mean