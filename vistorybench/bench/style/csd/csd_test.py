from .csd_model import CSD_CLIP
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def load_csd(w_path):
    csd_image_encoder = CSD_CLIP(only_global_token=True)
    state_dict = torch.load(w_path, map_location="cpu")
    csd_image_encoder.load_state_dict(state_dict, strict=False)
    
    # Freeze all parameters
    csd_image_encoder.eval()
    for param in csd_image_encoder.parameters():
        param.requires_grad = False
        
    # Ensure using float32 type
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
    """Return preprocessed tensor with gradients"""
    image_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_array).cuda()[None,...].permute(0,3,1,2)
    return tensor.requires_grad_(True)  # Important: enable gradients


def force_resize_to_224(image_tensor):
    """Force resize any input to 224x224"""
    return T.functional.resize(
        image_tensor,  # 自动转Tensor并加batch维度
        size=[224, 224], 
        interpolation=T.InterpolationMode.BILINEAR
    )

def get_csd_loss(style_img = None, render_img = None):
    # Initialize model
    csd_encoder = load_csd().cuda()

    # # Load and preprocess images
    # if style_img is None:
    #     style_img = Image.open('style_image.jpg').convert('RGB')
    # if render_img is None:
    #     render_img = Image.open('render_image.jpg').convert('RGB')

    # Style image doesn't need gradients, render image needs gradients
    if isinstance(style_img, (Image.Image)):
        print(f'Input CSD style image is an image, will convert to tensor next')
        style_tensor = preprocess(style_img).detach()  # detach to block gradients
    elif isinstance(style_img, (torch.Tensor)):
        style_tensor = style_img

    if isinstance(render_img, (Image.Image)):
        print(f'Input CSD render image is an image, will convert to tensor next')
        render_tensor = preprocess(render_img)         # Keep gradients
    elif isinstance(render_img, (torch.Tensor)):
        render_tensor = render_img

    style_tensor = force_resize_to_224(style_tensor) 

    print(f'style_tensor:{style_tensor.shape}')
    print(f'render_tensor:{render_tensor.shape}')

    # Forward computation
    style_embed = encode(style_tensor, csd_encoder)
    render_embed = encode(render_tensor, csd_encoder)

    print(f'style_embed:{style_embed.shape}')
    print(f'render_embed:{render_embed.shape}')

    # Calculate cosine similarity loss
    cos_sim = F.cosine_similarity(style_embed, render_embed, dim=-1)
    loss = 1 - cos_sim.mean()

    # Backpropagation test
    # loss.backward()

    print(f'Initial loss value: {loss.item():.4f}')
    print(f'Whether render image gradient exists: {render_tensor.grad is not None}')
    if render_tensor.grad is not None:
        print(f'Gradient norm: {render_tensor.grad.norm().item():.4f}')

    return loss



def get_csd_score(csd_encoder = None, img1_path = None, img2_path = None):
    # Initialize model
    if csd_encoder is None:
        csd_encoder = load_csd().cuda()

    # # Load and preprocess images
    # if img1 is None:
    #     img1 = Image.open('style_image.jpg').convert('RGB')
    # if img2 is None:
    #     img2 = Image.open('render_image.jpg').convert('RGB')

    Img1 = Image.open(img1_path).convert('RGB')
    image1_tensor = preprocess(Img1).detach() # detach to block gradients

    Img2 = Image.open(img2_path).convert('RGB')
    image2_tensor = preprocess(Img2).detach() # Keep gradients

    image1_tensor = force_resize_to_224(image1_tensor) 
    image2_tensor = force_resize_to_224(image2_tensor) 

    print(f'image1_tensor:{image1_tensor.shape}')
    print(f'image2_tensor:{image2_tensor.shape}')

    # Forward computation
    image1_embed = encode(image1_tensor, csd_encoder)
    image2_embed = encode(image2_tensor, csd_encoder)

    print(f'image1_embed:{image1_embed.shape}')
    print(f'image2_embed:{image2_embed.shape}')

    # Calculate cosine similarity loss
    cos_sim = F.cosine_similarity(image1_embed, image2_embed, dim=-1)
    cos_sim_mean = cos_sim.mean()

    return cos_sim, cos_sim_mean