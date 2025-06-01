import torch
import torch.nn as nn
import copy
from torch.autograd import Function
from .clip_model import build_model

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


## taken from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/modules.py
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def init_weights(m): # TODO: do we need init for layernorm?
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=1e-6)


class CSD_CLIP(nn.Module):
    """backbone + projection head"""
    def __init__(
            self,
            name='vit_large',
            only_global_token=True,
            multi_layer_feats=False,
            device='cuda',
            ):
        super().__init__()

        clipmodel = build_model(name=name, state_dict=None).to(device)
        self.backbone = clipmodel.visual

        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)

        self.backbone.proj = None
        self.backbone.only_global_token = only_global_token
        self.backbone.multi_layer_feats = multi_layer_feats

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def forward(self, input_data, alpha=None):
        
        if not self.backbone.multi_layer_feats:
            feature = self.backbone(input_data)

            if alpha is not None:
                reverse_feature = ReverseLayerF.apply(feature, alpha)
            else:
                reverse_feature = feature

            style_output = feature @ self.last_layer_style
            style_output = nn.functional.normalize(style_output, dim=2, p=2)

            content_output = reverse_feature @ self.last_layer_content
            content_output = nn.functional.normalize(content_output, dim=2, p=2)
        else:
            feature = None
            content_output = None
            style_output = self.backbone(input_data) # [B, L, 1024]
            
        return_dict = {
            'feature': feature, 
            'content': content_output, 
            'style': style_output}
        return return_dict
