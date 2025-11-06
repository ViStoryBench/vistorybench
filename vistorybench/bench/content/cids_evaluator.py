import os
import glob
from PIL import Image
import torch
import json
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.ops import box_convert
import insightface
from groundingdino.util.inference import load_model, load_image, predict, annotate
from facenet_pytorch import InceptionResnetV1
import sys
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import scipy.optimize
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

from vistorybench.bench.base_evaluator import BaseEvaluator
from vistorybench.dataset_loader.read_outputs import load_outputs

# Dynamically add AdaFace submodule to Python path to resolve imports
# within the submodule, as it's not a standard package.
adaface_path = os.path.join(os.path.dirname(__file__), 'AdaFace')
if adaface_path not in sys.path:
    sys.path.append(adaface_path)

from .AdaFace.inference import load_pretrained_model, to_input, adaface_models

class MultiModelFeatures:
    """Container class for saving multi-model features"""
    def __init__(self, features_dict):
        """
        Args:
            features_dict: Dictionary, keys are model names, values are feature tensors
        """
        self.features_dict = features_dict
        self.model_names = list(features_dict.keys())
        
    def to(self, device):
        """Move all features to specified device"""
        new_dict = {}
        for name, feat in self.features_dict.items():
            if feat is not None:
                new_dict[name] = feat.to(device)
            else:
                new_dict[name] = feat
        return MultiModelFeatures(new_dict)
        
    def cuda(self):
        """Move all features to GPU"""
        return self.to('cuda')
        
    def cpu(self):
        """Move all features to CPU"""
        return self.to('cpu')
        
    def __getitem__(self, key):
        return self.features_dict[key]
        
    def keys(self):
        return self.features_dict.keys()
        
    def values(self):
        return self.features_dict.values()
    
    def get_concatenated_features(self):
        """Get concatenated features"""
        valid_features = [feat for feat in self.features_dict.values() if feat is not None]
        if valid_features:
            # Ensure all features are 2-dimensional
            normalized_features = []
            for feat in valid_features:
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                # Normalize features
                feat = F.normalize(feat, p=2, dim=1)
                normalized_features.append(feat)
            
            # Concatenate features
            if len(normalized_features) > 1:
                # Take the first row (first face)
                face_features = [feat[0] for feat in normalized_features]
                combined_feature = torch.cat(face_features, dim=0)
            else:
                combined_feature = normalized_features[0][0]
                
            return combined_feature.unsqueeze(0)  # Return [1, combined_dim]
        return None
    
    def get_single_model_features(self, model_name):
        """Get features from a single model"""
        if model_name in self.features_dict and self.features_dict[model_name] is not None:
            feat = self.features_dict[model_name]
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            return F.normalize(feat, p=2, dim=1)
        return None

class CIDSEvaluator(BaseEvaluator):
    def __init__(self, config: dict, timestamp: str, mode: str, language: str, outputs_timestamp=None):
        super().__init__(config, timestamp, mode, language, outputs_timestamp)

        # Device
        self.device = torch.device(self.get_device())

        # Defaults
        self.BOX_TRESHOLD = 0.25
        self.TEXT_TRESHOLD = 0.25
        self.PRE_TEXT_PROMPT = {}

        # Evaluator-specific config
        self.cids_cfg = self.get_evaluator_config('cids')
        self.ref_mode = self.cids_cfg.get('ref_mode', 'origin')
        self.use_multi_face_encoder = self.cids_cfg.get('use_multi_face_encoder', False)
        self.ensemble_method = self.cids_cfg.get('ensemble_method', 'average')
        
        dino = self.cids_cfg.get('detection', {}).get('dino', {})
        if 'box_threshold' in dino:
            self.BOX_TRESHOLD = dino['box_threshold']
        if 'text_threshold' in dino:
            self.TEXT_TRESHOLD = dino['text_threshold']

        match = self.cids_cfg.get('matching', {})
        self.superfluous_threshold = match.get('superfluous_threshold', 0.5)
        self.topk_per_nochar = match.get('topk_per_nochar', 3)
        self.ensemble_weights = self.cids_cfg.get('ensemble_weights', {})
        # Copy-paste metric parameters for chordal metric (arccos-free, range [0,1])
        self.copy_paste_eps = float(self.cids_cfg.get('copy_paste_eps', 1e-6))
        self.copy_paste_tau_chord = float(self.cids_cfg.get('copy_paste_tau_chord', 1e-6))

        # ArcFace detection thresholds stepping (constants)
        self.arcface_det_thresh_initial = 0.45
        self.arcface_det_thresh_min = 0.1
        self.arcface_det_thresh_step = 0.1

        # Cosine similarity function
        self.cos = F.cosine_similarity

        # Prompt Align GPT-V config for single_character_action integration
        pa_cfg = self.get_evaluator_config('prompt_align')
        gpt_cfg = pa_cfg.get('gpt', {}) if isinstance(pa_cfg, dict) else {}
        model = self.get_cli_arg('model_id') or gpt_cfg.get('model') or 'gpt-4o'
        base_url = gpt_cfg.get('base_url') or self.get_base_url()
        api_key = gpt_cfg.get('api_key') or self.get_api_key()
        self.gpt_api_pkg = (model, api_key, base_url)
        self.gpt_workers = int(gpt_cfg.get('workers', 1) or 1)
        self._pa_skip_warned = False
        self._pa_prompt_path = 'vistorybench/bench/prompt_align/user_prompts/user_prompt_single_character_text_align.txt'

        # Load models on initialization
        self._load_models(self.pretrain_path, self.device)

    def _load_models(self, pretrain_path, device='cuda'):
        """Load all models required for CIDS evaluation."""
        pretrain_path = Path(pretrain_path)

        # Load GroundingDINO
        try:
            # Correctly reference the config within the bench/content directory
            gd_config = os.path.join(os.path.dirname(__file__), 'GroundingDINO_SwinT_OGC.py')
            gd_weights = pretrain_path / 'groundingdino/weights/groundingdino_swint_ogc.pth'
            self.dino = load_model(str(gd_config), str(gd_weights)).to(device)
            print("GroundingDINO model loaded successfully")
        except Exception as e:
            print(f"Could not load GroundingDINO model: {e}")
            self.dino = None

        # Load CLIP
        try:
            clip_model_id = "openai/clip-vit-large-patch14"
            if isinstance(self.cids_cfg, dict):
                enc = self.cids_cfg.get('encoders')
                if isinstance(enc, dict) and 'clip' in enc and 'model_id' in enc['clip']:
                    clip_model_id = enc['clip']['model_id']
            self.clip = CLIPModel.from_pretrained(clip_model_id).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Could not load CLIP model: {e}")
            self.clip = None
            self.clip_processor = None

        # Load ArcFace
        try:
            arc_name = "antelopev2"
            arc_providers = ['CUDAExecutionProvider']
            ctx_id = 0
            det0 = self.arcface_det_thresh_initial
            self.arcface = insightface.app.FaceAnalysis(root=pretrain_path / 'insightface', name=arc_name, providers=arc_providers)
            self.arcface.prepare(ctx_id=ctx_id, det_thresh=det0)
            print("ArcFace model loaded successfully")
        except Exception as e:
            print(f"Could not load ArcFace model: {e}")
            self.arcface = None

        # Load FaceNet
        try:
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            print("FaceNet model loaded successfully")
        except Exception as e:
            print(f"Could not load FaceNet model: {e}")
            self.facenet = None

        # Load AdaFace
        try:
            model_key = 'ir_101'
            ada_ckpt = str(pretrain_path / 'adaface/adaface_ir101_webface12m.ckpt')
            adaface_models[model_key] = ada_ckpt
            self.adaface = load_pretrained_model(model_key).to(device)
            print("AdaFace model loaded successfully")
        except Exception as e:
            print(f"Could not load AdaFace model: {e}")
            self.adaface = None

        # Initialize FaceRestoreHelper
        try:
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=112,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='jpg',
                use_parse=False
            )
            print("Face helper initialized successfully")
        except Exception as e:
            print(f"Could not initialize Face helper: {e}")
            self.face_helper = None

    def _get_encoder_name(self, char_tag):
        """Select appropriate encoder based on character tag and configuration"""
        if char_tag == "realistic_human":
            return "face_mix" if self.use_multi_face_encoder else "arcface"
        else:
            return "clip"

    def _get_arcface_features(self, img_np, det_thresh=None):
        """Extract facial features using ArcFace model"""
        curr_thresh = det_thresh if det_thresh is not None else self.arcface_det_thresh_initial
        while curr_thresh >= self.arcface_det_thresh_min:
            self.arcface.prepare(ctx_id=0, det_thresh=curr_thresh)
            faces = self.arcface.get(img_np)
            if len(faces) > 0:
                return torch.from_numpy(faces[0].embedding)
            curr_thresh -= self.arcface_det_thresh_step
        return None

    def _get_adaface_features(self, img_pil):
        """Extract facial features using AdaFace model"""
        if self.face_helper is None:
            return None
            
        try:
            self.face_helper.clean_all()
            img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            self.face_helper.read_image(img_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False)
            self.face_helper.align_warp_face()

            aligned_faces = self.face_helper.cropped_faces
            if len(aligned_faces) > 0:
                aligned_face = aligned_faces[0]  # Take the first face
                aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                bgr_input = to_input(aligned_rgb)
                bgr_input = bgr_input.to(self.device)
                
                with torch.no_grad():
                    feature, _ = self.adaface(bgr_input)
                return feature.detach().cpu().squeeze()
        except Exception as e:
            print(f"AdaFace feature extraction failed: {e}")
        return None

    def _get_facenet_features(self, img_pil):
        """Extract facial features using FaceNet model"""
        if self.face_helper is None:
            return None
            
        try:
            self.face_helper.clean_all()
            img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            self.face_helper.read_image(img_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False)
            self.face_helper.align_warp_face()

            aligned_faces = self.face_helper.cropped_faces
            if len(aligned_faces) > 0:
                aligned_face = aligned_faces[0]  # Take the first face
                aligned_face = torch.tensor(aligned_face).float() / 255.0
                aligned_face = aligned_face.permute(2, 0, 1)
                
                with torch.no_grad():
                    feature = self.facenet(aligned_face.unsqueeze(0).to(self.device))
                return feature.detach().cpu().squeeze()
        except Exception as e:
            print(f"FaceNet feature extraction failed: {e}")
        return None

    def get_char_feat(self, img: Image.Image | list[Image.Image], encoder_name="arcface", det_thresh = None) -> torch.Tensor:
        if not isinstance(img, list):
            img = [img]

        if len(img) == 0:
            print("\033[33mWarning: Empty image list provided to get_char_feat\033[0m")
            return None

        if encoder_name == "clip":
            inputs = self.clip_processor(images=img, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.clip.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)

        elif encoder_name == "arcface":
            curr_thresh = det_thresh if det_thresh is not None else self.arcface_det_thresh_initial
            image_features = []
            for _img in img:
                _img_np = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
                while(True):
                    if curr_thresh < self.arcface_det_thresh_min:
                        break
                    self.arcface.prepare(ctx_id=0, det_thresh=curr_thresh)
                    _img_faces = self.arcface.get(_img_np)
                    if len(_img_faces) > 0:
                        image_features.append(torch.from_numpy(_img_faces[0].embedding))
                        break
                    else:
                        _img.save(f"{self.mid_result_dir}/bad_case.png")
                        print(f"No face detected, auto re-try, curr_thresh: {curr_thresh}")
                        curr_thresh -= self.arcface_det_thresh_step
            if image_features:
                image_features = torch.stack(image_features, dim=0)
                image_features = F.normalize(image_features, p=2, dim=1)
            else:
                image_features = None

        elif encoder_name == "face_mix":
            all_multi_features = []
            
            for _img in img:
                _img_np = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
                model_features = {}
                
                arcface_features = self._get_arcface_features(_img_np, det_thresh)
                if arcface_features is not None:
                    if arcface_features.dim() == 1:
                        arcface_features = arcface_features.unsqueeze(0)
                    model_features['arcface'] = F.normalize(arcface_features, p=2, dim=1)
                
                if self.adaface is not None and self.face_helper is not None:
                    adaface_features = self._get_adaface_features(_img)
                    if adaface_features is not None:
                        if adaface_features.dim() == 1:
                            adaface_features = adaface_features.unsqueeze(0)
                        model_features['adaface'] = F.normalize(adaface_features, p=2, dim=1)
                
                if self.facenet is not None and self.face_helper is not None:
                    facenet_features = self._get_facenet_features(_img)
                    if facenet_features is not None:
                        if facenet_features.dim() == 1:
                            facenet_features = facenet_features.unsqueeze(0)
                        model_features['facenet'] = F.normalize(facenet_features, p=2, dim=1)
                
                if model_features:
                    multi_feat = MultiModelFeatures(model_features)
                    all_multi_features.append(multi_feat)
            
            if all_multi_features:
                image_features = all_multi_features
            else:
                image_features = None
        else:
            raise NotImplementedError

        return image_features

    def _compute_ensemble_similarity(self, multi_feat1, multi_feat2, method="average"):
        if not isinstance(multi_feat1, MultiModelFeatures) or not isinstance(multi_feat2, MultiModelFeatures):
            if hasattr(multi_feat1, 'get_concatenated_features'):
                feat1 = multi_feat1.get_concatenated_features()
            else:
                feat1 = multi_feat1
            if hasattr(multi_feat2, 'get_concatenated_features'):
                feat2 = multi_feat2.get_concatenated_features()
            else:
                feat2 = multi_feat2
                
            if feat1 is None or feat2 is None:
                return 0.0
            return F.cosine_similarity(feat1, feat2, dim=1).item()
        
        if method == "concatenate":
            feat1 = multi_feat1.get_concatenated_features()
            feat2 = multi_feat2.get_concatenated_features()
            if feat1 is None or feat2 is None:
                return 0.0
            return F.cosine_similarity(feat1, feat2, dim=1).item()
            
        elif method == "average":
            similarities = []
            common_models = set(multi_feat1.keys()) & set(multi_feat2.keys())
            
            for model_name in common_models:
                feat1 = multi_feat1[model_name]
                feat2 = multi_feat2[model_name]
                if feat1 is not None and feat2 is not None:
                    sim_matrix = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2)
                    sim_matrix_np = sim_matrix.detach().cpu().numpy()
                    cost_matrix = -sim_matrix_np
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                    
                    matched_scores = sim_matrix_np[row_ind, col_ind]
                    avg_sim = matched_scores.mean()
                    similarities.append(avg_sim)
            
            if similarities:
                return sum(similarities) / len(similarities)
            else:
                return 0.0
                
        elif method == "weighted":
            similarities = []
            weights = []
            common_models = set(multi_feat1.keys()) & set(multi_feat2.keys())
            
            model_weights = self.ensemble_weights
            
            for model_name in common_models:
                feat1 = multi_feat1[model_name]
                feat2 = multi_feat2[model_name]
                if feat1 is not None and feat2 is not None:
                    sim_matrix = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2)
                    sim_matrix_np = sim_matrix.detach().cpu().numpy()
                    cost_matrix = -sim_matrix_np
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                    
                    matched_scores = sim_matrix_np[row_ind, col_ind]
                    avg_sim = matched_scores.mean()
                    weight = model_weights.get(model_name, 1.0)
                    similarities.append(avg_sim * weight)
                    weights.append(weight)
            
            if similarities and weights:
                return sum(similarities) / sum(weights)
            else:
                return 0.0
        else:
            return self._compute_ensemble_similarity(multi_feat1, multi_feat2, "average")
    
    def _compute_multimodel_similarity_matrix(self, output_feats, ref_feats, method="average"):
        is_multi_output = isinstance(output_feats, list) and len(output_feats) > 0 and isinstance(output_feats[0], MultiModelFeatures)
        is_multi_ref = isinstance(ref_feats, list) and len(ref_feats) > 0 and isinstance(ref_feats[0], MultiModelFeatures)
        
        if is_multi_output or is_multi_ref:
            if is_multi_output and not is_multi_ref:
                similarities = []
                for out_feat in output_feats:
                    ref_multi = MultiModelFeatures({'single': ref_feats})
                    sim = self._compute_ensemble_similarity(out_feat, ref_multi, method)
                    similarities.append([sim])
                return torch.tensor(similarities)
                
            elif not is_multi_output and is_multi_ref:
                similarities = []
                out_multi = MultiModelFeatures({'single': output_feats})
                for ref_feat in ref_feats:
                    sim = self._compute_ensemble_similarity(out_multi, ref_feat, method)
                    similarities.append([sim])
                return torch.tensor(similarities).T
                
            elif is_multi_output and is_multi_ref:
                similarities = []
                for out_feat in output_feats:
                    row_sims = []
                    for ref_feat in ref_feats:
                        sim = self._compute_ensemble_similarity(out_feat, ref_feat, method)
                        row_sims.append(sim)
                    similarities.append(row_sims)
                return torch.tensor(similarities)
        
        if not isinstance(output_feats, torch.Tensor):
            output_feats = torch.tensor(output_feats)
        if not isinstance(ref_feats, torch.Tensor):
            ref_feats = torch.tensor(ref_feats)
        return (output_feats @ ref_feats.T)

    def dino_detect(self, inp_img: str, inp_cap: str, box_threshold=None, text_threshold=None):
        if box_threshold is None:
            box_threshold = self.BOX_TRESHOLD
        if text_threshold is None:
            text_threshold = self.TEXT_TRESHOLD
        
        try:
            if not os.path.exists(inp_img):
                print(f"\033[31mError: Image file not found for DINO detection: {inp_img}\033[0m")
                empty_boxes = torch.empty(0, 4)
                empty_logits = torch.empty(0)
                empty_phrases = []
                return empty_boxes, empty_logits, empty_phrases, None
                
            image_source, image = load_image(inp_img)
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.dino,
                    image=image,
                    caption=inp_cap,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            return boxes, logits, phrases, annotated_frame
        except Exception as e:
            print(f"\033[31mError in DINO detection for {inp_img}: {str(e)}\033[0m")
            empty_boxes = torch.empty(0, 4)
            empty_logits = torch.empty(0)
            empty_phrases = []
            return empty_boxes, empty_logits, empty_phrases, None

    def crop_img(self, img_src: str, boxes) -> list[Image.Image]:
        try:
            if not os.path.exists(img_src):
                print(f"\033[31mError: Image file not found for cropping: {img_src}\033[0m")
                return []
            
            if len(boxes) == 0:
                return []
                
            img_source = Image.open(img_src)
            w, h = img_source.size
            boxes = (boxes * torch.Tensor([w, h, w, h])).int()
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            cropped_images = []
            for box in xyxy:
                try:
                    x_min, y_min, x_max, y_max = box.tolist()
                    x_min = max(0, min(x_min, w))
                    y_min = max(0, min(y_min, h))
                    x_max = max(0, min(x_max, w))
                    y_max = max(0, min(y_max, h))
                    
                    if x_max > x_min and y_max > y_min:
                        cropped = img_source.crop((x_min, y_min, x_max, y_max))
                        cropped_images.append(cropped)
                except Exception as e:
                    print(f"\033[33mWarning: Failed to crop box {box}: {str(e)}\033[0m")
                    continue
            return cropped_images
        except Exception as e:
            print(f"\033[31mError in crop_img for {img_src}: {str(e)}\033[0m")
            return []

    def evaluate(self, method: str, story_id: str, **kwargs):

        # Create a directory for intermediate results (aligned to unified result schema)
        run_dir = os.path.join(self.result_path, method, self.mode, self.language, self.timestamp)
        self.mid_result_dir = os.path.join(run_dir, 'metrics', 'cids', 'mid_results', story_id)
        os.makedirs(self.mid_result_dir, exist_ok=True)

        # GPT-V utils import (local) for single_character_action scoring
        from vistorybench.bench.prompt_align.gptv_utils import gptv_query, load_img_content
        story_data = self.story_dataset.load_story(story_id)
        shots = story_data['shots']
        Characters = story_data['characters']

        all_outputs = load_outputs(
            outputs_root=self.output_path,
            methods=[method],
            modes=[self.mode],
            languages=[self.language],
            timestamps=[self.outputs_timestamp],
            return_latest=False
        )
        story_outputs = all_outputs.get(story_id)
        if not story_outputs:
            print(f"Warning: No outputs found for story {story_id}, method {method}")
            return None

        CHARACTER = list(Characters.keys())
        REF_PATH = {char: os.path.join(self.dataset_path, 'ViStory', story_id, "image", char) for char in CHARACTER}
        TEXT_PROMPT = {char: "character" for char in CHARACTER} if not self.PRE_TEXT_PROMPT else self.PRE_TEXT_PROMPT

        if self.ref_mode == 'mid-gen':
            MID_GEN_REF_PATH = {char: None for char in CHARACTER}
            MID_GEN_CHAR_LIST = story_outputs.get('chars', [])

        # Load references
        ref_clip_feats = {}
        for char in CHARACTER:
            enc_name = self._get_encoder_name(Characters[char]["tag"])
            ch_name = Characters[char]['name']
            input_ref_imgs = []

            if self.ref_mode == 'origin':
                ref_imgs = sorted(glob.glob(os.path.join(REF_PATH[char], '**/*.jpg'), recursive=True))
            elif self.ref_mode == 'mid-gen':
                ref_imgs = []
                for img_path in MID_GEN_CHAR_LIST:
                    char_file_name = os.path.basename(img_path).split(".")[0]
                    if char_file_name in MID_GEN_REF_PATH.keys() and char == char_file_name:
                        MID_GEN_REF_PATH[char] = img_path
                        break
                    elif ch_name in char_file_name:
                        MID_GEN_REF_PATH[char] = img_path
                        break
                if MID_GEN_REF_PATH[char] is None:
                    print(f'[WARN] Missing character: {char}, Assign an empty value [] (available character: {CHARACTER})')
                else:
                    ref_imgs = sorted(glob.glob(MID_GEN_REF_PATH[char], recursive=True))

            for img in ref_imgs:
                boxes, logits, _, _ = self.dino_detect(img, TEXT_PROMPT[char])
                if len(logits) == 0:
                    print(f"\033[33mNo objects detected in reference image {img} for {char} with prompt: {TEXT_PROMPT[char]}\033[0m")
                    continue
                _, indices = torch.topk(logits, 1)
                boxes = boxes[indices]
                cropped_imgs = self.crop_img(img, boxes)
                if len(cropped_imgs) != 0:
                    input_ref_imgs.append(cropped_imgs[0])
                else:
                    print(f"\033[33mNo char: {char} found in {img} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")

            if len(input_ref_imgs) > 0:
                ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
                if ref_feats is None:
                    input_ref_imgs = [Image.open(x) for x in ref_imgs]
                    if len(input_ref_imgs) > 0:
                        ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
            else:
                print(f"\033[33mNo valid cropped reference images for {char}, trying original images\033[0m")
                input_ref_imgs = [Image.open(x) for x in ref_imgs]
                if len(input_ref_imgs) > 0:
                    ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
                else:
                    ref_feats = None

            assert ref_feats is not None, f"Cant get ref char: {char}, please check. No valid reference images found."
            ref_clip_feats[char] = ref_feats

        results = {"cids": {}}
        char_pil_imgs = {}
        occm_scores = []
        new_shot_indices = []

        # Single-character action alignment (per-shot, per-character)
        pa_detailed_scores = {}
        pa_story_scores = []
        model_type, api_key, base_url = self.gpt_api_pkg if hasattr(self, 'gpt_api_pkg') else (None, '', '')
        pa_available = bool(api_key) and bool(base_url)
        if not pa_available and not getattr(self, '_pa_skip_warned', False):
            print("\033[33mWarning: single_character_action skipped: missing GPT-V base_url or api_key in config.\033[0m")
            self._pa_skip_warned = True
        single_prompt_text = None
        if pa_available:
            try:
                with open(self._pa_prompt_path, 'r') as _pf:
                    single_prompt_text = _pf.read()
            except Exception as _e:
                print(f"\033[33mWarning: failed to read single_character_action prompt file: {_e}\033[0m")
                pa_available = False


        for shot in shots:
            shot_results = {}
            pa_char_scores = {}
            shot_id = int(shot["index"]) - 1

            if shot_id >= len(story_outputs['shots']):
                print(f"Warning: shot_id {shot_id} is out of bounds. Skipping...")
                continue

            target_img_path = story_outputs['shots'][shot_id]

            if not os.path.exists(target_img_path):
                print(f"\033[31mError: Image file not found: {target_img_path}. Skipping shot {shot_id}.\033[0m")
                for char in shot['character_key']:
                    shot_results.update({char: {"box": "null", "cross_sim": 0.0}})
                E = len(shot['character_key']); D = 0; epsilon = 1e-6
                occm_scores.append(100 * np.exp(-float(abs(D - E)) / (epsilon + E)))
                results.update({f"shot-{shot_id}": shot_results})
                continue

            # Fresh computation for new shot
            for char in shot['character_key']:
                enc_name = self._get_encoder_name(Characters[char]["tag"])
                if char not in char_pil_imgs:
                    char_pil_imgs[char] = []
                boxes, logits, phrases, annotated_frame = self.dino_detect(target_img_path, TEXT_PROMPT[char])

                if len(logits) == 0:
                    shot_results.update({char: {"box": "null", "cross_sim": 0.0}})
                    continue

                _, indices = torch.topk(logits, min(len(logits), len(shot['character_key'])))
                boxes = boxes[indices]
                cropped_imgs = self.crop_img(target_img_path, boxes)
                if len(cropped_imgs) != 0:
                    output_feats = self.get_char_feat(cropped_imgs, encoder_name=enc_name)
                    if output_feats is None:
                        shot_results.update({char: {"box": "null", "cross_sim": 0.0}})
                        continue

                    cosine_sims = {}
                    for _char in shot['character_key']:
                        similarity_matrix = self._compute_multimodel_similarity_matrix(
                            output_feats.cuda() if not isinstance(output_feats, list) else output_feats,
                            ref_clip_feats[_char].cuda() if not isinstance(ref_clip_feats[_char], list) else ref_clip_feats[_char],
                            method=self.ensemble_method
                        )
                        cosine_sims.update({_char: similarity_matrix.max(dim=1).values.cpu()})

                    if char not in cosine_sims.keys():
                        continue

                    sims_bak = cosine_sims[char].tolist()
                    available_indices = list(range(len(sims_bak)))
                    matched_flag = False

                    while len(sims_bak) > 0:
                        local_id = torch.argmax(torch.Tensor(sims_bak)).item()
                        original_id = available_indices[local_id]

                        conflict_found = False
                        for _char in shot['character_key']:
                            if _char != char:
                                if (original_id < len(cosine_sims[_char]) and
                                    original_id < len(cosine_sims[char]) and
                                    cosine_sims[_char][original_id] > cosine_sims[char][original_id]):
                                    conflict_found = True
                                    break

                        if not conflict_found:
                            matched_flag = True
                            boxes_id = original_id
                            break

                        sims_bak.pop(local_id)
                        available_indices.pop(local_id)

                    if matched_flag:
                        boxes_to_write = [round(x, 3) for x in boxes[boxes_id].tolist()]
                        try:
                            matched_output_feat = output_feats[boxes_id:boxes_id+1] if not isinstance(output_feats, list) else [output_feats[boxes_id]]
                            cross_sim_matrix = self._compute_multimodel_similarity_matrix(
                                matched_output_feat,
                                ref_clip_feats[char],
                                method=self.ensemble_method
                            )
                            shot_cross_sim = round(cross_sim_matrix.mean().item(), 4) if cross_sim_matrix.numel() > 0 else 0.0
                        except Exception as e:
                            print(f"\033[33mWarning: Failed to compute cross_sim for {char} in shot {shot_id}: {str(e)}\033[0m")
                            shot_cross_sim = 0.0

                        shot_results.update({char: {"box": boxes_to_write, "cross_sim": shot_cross_sim}})
                        char_pil_imgs[char].append(cropped_imgs[boxes_id])
                        cropped_imgs[boxes_id].save(f"{self.mid_result_dir}/shot{shot_id:02d}-{char}.png")

                        # Single-character action alignment via GPT-V (only for matched characters)
                        if pa_available and single_prompt_text:
                            try:
                                # Fallback to closest action/script field
                                prompt_text = shot.get('script') or shot.get('character_action') or shot.get('action') or shot.get('description') or ''
                                # Compose user content
                                user_content = [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "text", "text": f"Evaluated Character Name is {Characters[char]['name']}"},
                                    load_img_content(cropped_imgs[boxes_id], image_mode='pil')
                                ]
                                transcript = [{"role":"system","content":[{"type": "text", "text": single_prompt_text}]},{"role": "user", "content": user_content}]
                                # Local retry with temperature escalation
                                import re
                                max_retry = 10
                                temp_start = 0.0
                                score_val = 0
                                while max_retry > 0:
                                    try:
                                        response = gptv_query(
                                            transcript,
                                            top_p=0.2,
                                            temp=temp_start,
                                            model_type=model_type,
                                            api_key=api_key,
                                            base_url=base_url,
                                        )
                                        pattern = r"(score|Score):\s*[a-zA-Z]*\s*(\d+)"
                                        matches = re.findall(pattern, response)
                                        if matches:
                                            score_val = int(matches[0][1])
                                            break
                                        else:
                                            temp_start += 0.1
                                            max_retry -= 1
                                    except Exception as e:
                                        print(f"\033[33mWarning: single_character_action scoring failed for story {story_id}, shot {shot_id}, char {char}: {e}\033[0m")
                                        temp_start += 0.1
                                        max_retry -= 1
                                # Record character-level score using real character name
                                pa_char_scores[Characters[char]['name']] = int(score_val)
                            except Exception as _e:
                                print(f"\033[33mWarning: single_character_action internal error for story {story_id}, shot {shot_id}, char {char}: {_e}\033[0m")
                    else:
                        shot_results.update({char: {"box": "null", "cross_sim": 0.0}})

                else:
                    shot_results.update({char: {"box": "null", "cross_sim": 0.0}})

            # OCCM for new shot
            E = len(shot['character_key'])
            try:
                D = sum(1 for _ch in shot['character_key']
                        if isinstance(shot_results.get(_ch, {}), dict) and shot_results[_ch].get("box") != "null")
            except Exception:
                D = 0
            epsilon = 1e-6
            occm_scores.append(100 * np.exp(-float(abs(D - E)) / (epsilon + E)))

            # Aggregate single_character_action for this shot
            if pa_available:
                if pa_char_scores:
                    shot_pa_mean = int(round(sum(pa_char_scores.values()) / len(pa_char_scores)))
                else:
                    shot_pa_mean = 0
                pa_detailed_scores[shot_id] = {
                    "single_character_action": shot_pa_mean,
                    "single_character_action_char": pa_char_scores.copy()
                }
                pa_story_scores.append(shot_pa_mean)

            results.update({f"shot-{shot_id}": shot_results})
            new_shot_indices.append(shot_id)

        # Aggregate per-character and story metrics using char_pil_imgs (from processed + new)
        copy_paste_cnt = 0
        shot_copy_paste_score = 0.0
        for char in CHARACTER:
            enc_name = self._get_encoder_name(Characters[char]["tag"])
            if char in char_pil_imgs and len(char_pil_imgs[char]) > 0:
                char_feats = self.get_char_feat(char_pil_imgs[char], encoder_name=enc_name)
                if char_feats is None:
                    results["cids"].update({char: {"cross": 0.0, "self": 0.0}})
                    continue

                cross_similarity_matrix = self._compute_multimodel_similarity_matrix(char_feats, ref_clip_feats[char], method=self.ensemble_method)
                cross_sim = cross_similarity_matrix.mean().item()

                self_similarity_matrix = self._compute_multimodel_similarity_matrix(char_feats, char_feats, method=self.ensemble_method)
                if self_similarity_matrix.shape[0] > 1:
                    indices = torch.triu_indices(self_similarity_matrix.shape[0], self_similarity_matrix.shape[1], offset=1)
                    self_sim = self_similarity_matrix[indices[0], indices[1]].mean().item()
                else:
                    self_sim = 1.0
                results["cids"].update({char: {"cross": round(cross_sim, 4), "self": round(self_sim, 4)}})

                ref_count = len(ref_clip_feats[char]) if isinstance(ref_clip_feats[char], list) else ref_clip_feats[char].shape[0]
                if ref_count > 1:
                    copy_paste_cnt += 1
                    cross_sims = self._compute_multimodel_similarity_matrix(
                        char_feats, ref_clip_feats[char], method=self.ensemble_method
                    )  # [G, R]
                    if cross_sims.shape[1] > 1:
                        # CopyChord/CopyRate (chordal metric, arccos-free, range [0,1])
                        # t = ref[0], r = ref[1:]
                        # Stabilize cosine to [-1, 1]
                        s_gt = torch.clamp(cross_sims[:, 0], -1.0, 1.0)       # [G]
                        s_gr = torch.clamp(cross_sims[:, 1:], -1.0, 1.0)       # [G, R-1]

                        # Reference to reference similarity to compute s_tr
                        ref_ref_sims = self._compute_multimodel_similarity_matrix(
                            ref_clip_feats[char], ref_clip_feats[char], method=self.ensemble_method
                        )  # [R, R]
                        # Here R >= 2 because cross_sims.shape[1] > 1
                        s_tr = torch.clamp(ref_ref_sims[0, 1:], -1.0, 1.0)     # [R-1]

                        # Chordal distances on unit sphere
                        d_gt = torch.sqrt(torch.clamp(2.0 * (1.0 - s_gt), min=eps))
                        d_gr = torch.sqrt(torch.clamp(2.0 * (1.0 - s_gr), min=eps))
                        d_tr = torch.sqrt(torch.clamp(2.0 * (1.0 - s_tr), min=eps))

                        # Stability parameters
                        eps = float(getattr(self, "copy_paste_eps", 1e-6))
                        tau = float(getattr(self, "copy_paste_tau_chord", 1e-6))

                        # Normalize by separation between t and r
                        denom = torch.clamp(d_tr, min=eps).unsqueeze(0)         # [1, R-1]
                        copy_chord = (d_gt.unsqueeze(1) - d_gr) / denom         # [G, R-1]
                        copy_rate = 0.5 * (1.0 + copy_chord)                    # [G, R-1]

                        # Degenerate case: if t and r are nearly identical, result is set to 0.5
                        if torch.any(d_tr < tau):
                            deg_cols = (d_tr < tau).nonzero(as_tuple=False).squeeze(-1)  # [K]
                            if deg_cols.numel() > 0:
                                copy_rate[:, deg_cols] = 0.5

                        char_copy_rate = copy_rate.mean().item()
                        shot_copy_paste_score += char_copy_rate
                    else:
                        copy_paste_cnt -= 1
            else:
                results["cids"].update({char: {"cross": 0.0, "self": 0.0}})

        if copy_paste_cnt > 0:
            shot_copy_paste_score /= copy_paste_cnt
            results.update({"copy-paste-score": shot_copy_paste_score})
        else:
            results.update({"copy-paste-score": "null"})

        try:
            cids_map = results.get("cids", {}) if isinstance(results, dict) else {}
            self_vals = []
            cross_vals = []
            for _char, _rec in cids_map.items():
                if isinstance(_rec, dict):
                    sv = _rec.get("self")
                    cv = _rec.get("cross")
                    if isinstance(sv, (int, float)):
                        self_vals.append(float(sv))
                    if isinstance(cv, (int, float)):
                        cross_vals.append(float(cv))
            cids_self_mean = round(sum(self_vals) / len(self_vals), 4) if self_vals else 0.0
            cids_cross_mean = round(sum(cross_vals) / len(cross_vals), 4) if cross_vals else 0.0
            occm_mean = round(sum(occm_scores) / len(occm_scores), 4) if occm_scores else 0.0
            cp = results.get("copy-paste-score", None)
            copy_paste_score = None if cp == "null" else float(cp) if isinstance(cp, (int, float)) else None

            results["metrics"] = {
                "cids_self_mean": cids_self_mean,
                "cids_cross_mean": cids_cross_mean,
                "copy_paste_score": copy_paste_score if copy_paste_score is not None else 0.0,
                "occm": occm_mean
            }
            if pa_available and pa_story_scores:
                results["metrics"]["single_character_action"] = int(round(sum(pa_story_scores) / len(pa_story_scores)))
        except Exception as _e:
            print(f"\033[33mWarning: failed to compute CIDS story metrics for {story_id}: {_e}\033[0m")

        if pa_available:
            results["detailed_scores"] = pa_detailed_scores
        results["new_shot_indices"] = new_shot_indices
        print(f"CIDS evaluation complete for story: {story_id}")
        return results

    def build_item_records(self, method: str, story_id: str, story_result):
        items = []
        try:
            if isinstance(story_result, dict):
                new_only = set(story_result.get("new_shot_indices") or [])
                # Existing CIDS item records
                for k, shot_res in story_result.items():
                    if not isinstance(k, str) or not k.startswith("shot-"):
                        continue
                    try:
                        shot_idx = int(k.split("-")[1])
                    except Exception:
                        shot_idx = k
                    if new_only and shot_idx not in new_only:
                        continue
                    if isinstance(shot_res, dict):
                        for char_key, v in shot_res.items():
                            if not isinstance(v, dict):
                                continue
                            cross_val = v.get("cross_sim", 0.0)
                            box = v.get("box", None)
                            if cross_val is None:
                                continue
                            item = {
                                "metric": {"name": "cids", "submetric": "cross_sim"},
                                "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx, "character_key": char_key},
                                "value": cross_val,
                                "unit": "cosine_similarity",
                                "extras": {"box": box},
                                "status": "complete",
                            }
                            items.append(item)
                # New PromptAlign single_character_action item records per character (if available)
                detailed = story_result.get("detailed_scores")
                if isinstance(detailed, dict):
                    for shot_idx, dims in detailed.items():
                        char_map = (dims or {}).get("single_character_action_char")
                        if isinstance(char_map, dict):
                            for char_name, score in char_map.items():
                                item = {
                                    "metric": {"name": "prompt_align", "submetric": "single_character_action"},
                                    "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx, "character_name": char_name},
                                    "value": score,
                                    "status": "complete",
                                }
                                items.append(item)
        except Exception as _e:
            print(f"\033[33mWarning: build_item_records failed for CIDS story {story_id}: {_e}\033[0m")
        return items