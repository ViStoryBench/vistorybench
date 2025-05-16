import os
from collections import OrderedDict
from os import PathLike
from typing import Final, Optional, Union

import torch
import torch.nn as nn
from transformers import SiglipVisionConfig, SiglipVisionModel, logging
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutputWithNoAttention

logging.set_verbosity_error()

URL: Final[str] = "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/models/aesthetic_predictor_v2_5.pth"


class AestheticPredictorV2p5Head(nn.Module):
    def __init__(self, input_hidden_size: int) -> None:
        super().__init__()
        self.scoring_head = nn.Sequential(
            nn.Linear(input_hidden_size, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def norm_embedding(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    def forward(self, image_embeds: torch.Tensor, norm=False) -> torch.Tensor:
        if norm:
            image_embeds = self.norm_embedding(image_embeds)
        return self.scoring_head(image_embeds)


class AestheticPredictorV2p5Model(SiglipVisionModel):
    PATCH_SIZE = 14

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.layers = AestheticPredictorV2p5Head(config.hidden_size)
        self.post_init()

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = outputs.pooler_output
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        prediction = self.layers(image_embeds_norm)

        return ImageClassifierOutputWithNoAttention(
            loss=None,
            logits=prediction,
            hidden_states=image_embeds,
        )


def load_apv2p5_state_dict(predictor_name_or_path: str | PathLike | None = None, *args, **kwargs):
    if predictor_name_or_path is None or not os.path.exists(predictor_name_or_path):
        state_dict = torch.hub.load_state_dict_from_url(URL, map_location="cpu")
    else:
        state_dict = torch.load(predictor_name_or_path, map_location="cpu")

    assert isinstance(state_dict, OrderedDict)

    return state_dict
