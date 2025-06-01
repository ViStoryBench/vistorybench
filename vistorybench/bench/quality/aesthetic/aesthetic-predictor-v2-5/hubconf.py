from os import PathLike

from model import AestheticPredictorV2p5Head, AestheticPredictorV2p5Model, load_apv2p5_state_dict
from transformers import SiglipImageProcessor


def aesthetic_predictor_v2_5(
    predictor_name_or_path: str | PathLike | None = None,
    encoder_model_name: str = "/data/pretrain/google/siglip-so400m-patch14-384",
    *args,
    **kwargs,
) -> tuple[AestheticPredictorV2p5Model, SiglipImageProcessor]:
    model = AestheticPredictorV2p5Model.from_pretrained(
        encoder_model_name, *args, **kwargs, local_files_only=True)

    processor = SiglipImageProcessor.from_pretrained(
        encoder_model_name, *args, **kwargs, local_files_only=True)

    state_dict = load_apv2p5_state_dict(predictor_name_or_path)
    model.layers.load_state_dict(state_dict)
    model.eval()

    return model, processor


def aesthetic_predictor_v2_5_head(
    predictor_name_or_path: str | PathLike | None = None,
    input_hidden_size=1152,
) -> AestheticPredictorV2p5Head:
    apv2p5_head = AestheticPredictorV2p5Head(input_hidden_size)

    state_dict = load_apv2p5_state_dict(predictor_name_or_path)
    apv2p5_head.load_state_dict(state_dict)
    apv2p5_head.eval()

    return apv2p5_head
