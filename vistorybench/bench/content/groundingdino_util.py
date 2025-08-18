import torch
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda", text_encoder_type: str = ''):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.text_encoder_type = text_encoder_type
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model
