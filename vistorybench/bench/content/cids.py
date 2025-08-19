import os
import cv2
import glob
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.ops import box_convert

import insightface
from insightface.data import get_image as ins_get_image
from transformers import CLIPProcessor, CLIPModel
# from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import load_model, load_image, predict, annotate

# New import: multi-model support
from facenet_pytorch import MTCNN, InceptionResnetV1
FACENET_AVAILABLE = True

# Fix AdaFace import issue
import sys
adaface_path = os.path.join(os.path.dirname(__file__), 'AdaFace')
if adaface_path not in sys.path:
    sys.path.insert(0, adaface_path)

from .AdaFace.inference import load_pretrained_model, to_input, adaface_models
ADAFACE_AVAILABLE = True

from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
FACEXLIB_AVAILABLE = True

import scipy.optimize


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


class CIDS:
    def __init__(self, model_pkg, pretrain_path, 
                pre_fix, output_dir, save_dir, ref_mode, 
                use_multi_face_encoder=False, ensemble_method="average"):

        (clip_model, clip_processor, dino_model) = model_pkg
        # >>>>>>>> Modify here
        # paths
        # self.DINO_MODEL = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        # self.DINO_WEIGHTS = "/data/pretrain/groundingdino/weights/groundingdino_swint_ogc.pth"

        self.PRE_FIX = pre_fix  # Story level
        self.OUTPUT_DIR = output_dir  # Story level results
        self.SAVE_DIR = save_dir # Save path

        self.use_multi_face_encoder = use_multi_face_encoder  # Whether to use multi-model fusion
        self.ensemble_method = ensemble_method  # Multi-model ensemble method: "average", "concatenate", "weighted"

        self.ref_mode = ref_mode
        self.mid_result_dir = os.path.join(self.SAVE_DIR, "mid_result")
        os.makedirs(self.mid_result_dir, exist_ok=True)
        # thresholds
        self.BOX_TRESHOLD = 0.25
        self.TEXT_TRESHOLD = 0.25
        # [optional] character key word for dino
        self.PRE_TEXT_PROMPT = {}
        # <<<<<<<<

        # ############################################################################################################

        self.REF_DIR = os.path.join(self.PRE_FIX, "image")
        self.PROMPT_PATH = os.path.join(self.PRE_FIX, "story.json")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load models
        # self.clip_model = CLIPModel.from_pretrained("/data/pretrain/openai/clip-vit-base-patch16").to(self.device)
        # self.processor = CLIPProcessor.from_pretrained("/data/pretrain/openai/clip-vit-base-patch16")
        self.clip_model = clip_model.to(self.device)
        self.processor = clip_processor

        self.arcface_model = insightface.app.FaceAnalysis(root=f'{pretrain_path}/insightface', name="antelopev2", providers=['CUDAExecutionProvider'])
        self.arcface_model.prepare(ctx_id=0, det_thresh=0.45)

        # self.dino_model = load_model(self.DINO_MODEL, self.DINO_WEIGHTS).to(self.device)
        self.dino_model = dino_model.to(self.device)

        # New: multi-model support initialization
        self.facenet_model = None
        self.adaface_model = None
        self.face_helper = None

        print(f'FACENET_AVAILABLE:{FACENET_AVAILABLE}, ADAFACE_AVAILABLE:{ADAFACE_AVAILABLE}, FACEXLIB_AVAILABLE:{FACEXLIB_AVAILABLE}')
        
        # Initialize FaceNet model
        if FACENET_AVAILABLE:
            self.facenet_model = InceptionResnetV1(pretrained=f'vggface2').eval().to(self.device)
            print("FaceNet model loaded successfully")
                
        # Initialize AdaFace model
        if ADAFACE_AVAILABLE:
            adaface_models['ir_101']= f"{pretrain_path}/adaface/adaface_ir101_webface12m.ckpt"
            self.adaface_model = load_pretrained_model('ir_101').to(self.device)
            print("AdaFace model loaded successfully")
                
        # Initialize face processing helper
        if FACEXLIB_AVAILABLE:
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=112,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='jpg',
                use_parse=False
            )
            print("Face helper initialized successfully")
            
        # Cosine similarity function
        self.cos = F.cosine_similarity

    def _get_encoder_name(self, char_tag):
        """Select appropriate encoder based on character tag and configuration"""
        if char_tag == "realistic_human":
            return "face_mix" if self.use_multi_face_encoder else "arcface"
        else:
            return "clip"


    def load_shots(self, prompt_shot_path: str) -> list[dict]:
        with open(prompt_shot_path, 'r', encoding='utf-8-sig') as f:
            shots = json.load(f)
        return shots["Shots"], shots["Characters"]

    def get_char_feat(self, img: Image.Image | list[Image.Image], encoder_name="arcface", det_thresh = 0.45) -> torch.Tensor:
        if not isinstance(img, list):
            img = [img]

        # Check if there are valid image inputs
        if len(img) == 0:
            print("\033[33mWarning: Empty image list provided to get_char_feat\033[0m")
            return None

        if encoder_name == "clip":
            inputs = self.processor(images=img, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)

        elif encoder_name == "arcface":
            curr_thresh = det_thresh
            image_features = []
            for _img in img:
                _img_np = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
                while(True):
                    if curr_thresh < 0.1:
                        break
                    self.arcface_model.prepare(ctx_id=0, det_thresh=curr_thresh)
                    _img_faces = self.arcface_model.get(_img_np)
                    if len(_img_faces) > 0:
                        image_features.append(torch.from_numpy(_img_faces[0].embedding))
                        break
                    else:
                        _img.save(f"{self.mid_result_dir}/bad_case.png")
                        print(f"No face detected, auto re-try, curr_thresh: {curr_thresh}")
                        curr_thresh -= 0.1
            if image_features:
                image_features = torch.stack(image_features, dim=0)
                image_features = F.normalize(image_features, p=2, dim=1)
            else:
                image_features = None

        elif encoder_name == "face_mix":
            # Use multi-model combination: AdaFace + FaceNet + ArcFace
            all_multi_features = []
            
            for _img in img:
                _img_np = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
                
                # Extract features from each model separately
                model_features = {}
                
                # 1. ArcFace feature extraction
                arcface_features = self._get_arcface_features(_img_np, det_thresh)
                if arcface_features is not None:
                    if arcface_features.dim() == 1:
                        arcface_features = arcface_features.unsqueeze(0)
                    model_features['arcface'] = F.normalize(arcface_features, p=2, dim=1)
                
                # 2. AdaFace feature extraction
                if self.adaface_model is not None and self.face_helper is not None:
                    adaface_features = self._get_adaface_features(_img)
                    if adaface_features is not None:
                        if adaface_features.dim() == 1:
                            adaface_features = adaface_features.unsqueeze(0)
                        model_features['adaface'] = F.normalize(adaface_features, p=2, dim=1)
                
                # 3. FaceNet feature extraction
                if self.facenet_model is not None and self.face_helper is not None:
                    facenet_features = self._get_facenet_features(_img)
                    if facenet_features is not None:
                        if facenet_features.dim() == 1:
                            facenet_features = facenet_features.unsqueeze(0)
                        model_features['facenet'] = F.normalize(facenet_features, p=2, dim=1)
                
                # Create MultiModelFeatures object
                if model_features:
                    multi_feat = MultiModelFeatures(model_features)
                    all_multi_features.append(multi_feat)
            
            if all_multi_features:
                # Return list of MultiModelFeatures, this requires special handling
                image_features = all_multi_features
            else:
                image_features = None
        else:
            raise NotImplementedError

        return image_features









    def _get_arcface_features(self, img_np, det_thresh=0.45):
        """Extract facial features using ArcFace model"""
        curr_thresh = det_thresh
        while curr_thresh >= 0.1:
            self.arcface_model.prepare(ctx_id=0, det_thresh=curr_thresh)
            faces = self.arcface_model.get(img_np)
            if len(faces) > 0:
                return torch.from_numpy(faces[0].embedding)
            curr_thresh -= 0.1
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
                # Convert BGR to RGB and create AdaFace input
                aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                # Convert to AdaFace required input format
                bgr_input = to_input(aligned_rgb)
                # Ensure input tensor is on the correct device
                bgr_input = bgr_input.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    feature, _ = self.adaface_model(bgr_input)
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
                # Convert format: BGR -> RGB, normalize to [0,1], adjust dimensions
                aligned_face = torch.tensor(aligned_face).float() / 255.0
                aligned_face = aligned_face.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
                
                # Extract features
                with torch.no_grad():
                    feature = self.facenet_model(aligned_face.unsqueeze(0).to(self.device))
                return feature.detach().cpu().squeeze()
        except Exception as e:
            print(f"FaceNet feature extraction failed: {e}")
        return None
        
    def _compute_ensemble_similarity(self, multi_feat1, multi_feat2, method="average"):
        """Calculate ensemble similarity of multi-model features
        
        Args:
            multi_feat1, multi_feat2: MultiModelFeatures objects or regular tensors
            method: ensemble method ("average", "concatenate", "weighted")
        """
        # If input is regular tensor, directly calculate cosine similarity
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
            # Concatenate all model features then calculate similarity
            feat1 = multi_feat1.get_concatenated_features()
            feat2 = multi_feat2.get_concatenated_features()
            if feat1 is None or feat2 is None:
                return 0.0
            return F.cosine_similarity(feat1, feat2, dim=1).item()
            
        elif method == "average":
            # Calculate similarity for each model feature separately then take average
            similarities = []
            common_models = set(multi_feat1.keys()) & set(multi_feat2.keys())
            
            for model_name in common_models:
                feat1 = multi_feat1[model_name]
                feat2 = multi_feat2[model_name]
                if feat1 is not None and feat2 is not None:
                    # Use Hungarian algorithm to find best match
                    sim_matrix = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2)
                    sim_matrix_np = sim_matrix.detach().cpu().numpy()
                    cost_matrix = -sim_matrix_np  # Convert to minimization problem
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                    
                    # Extract matched similarity scores
                    matched_scores = sim_matrix_np[row_ind, col_ind]
                    avg_sim = matched_scores.mean()
                    similarities.append(avg_sim)
            
            if similarities:
                return sum(similarities) / len(similarities)
            else:
                return 0.0
                
        elif method == "weighted":
            # Weighted average (weights can be set based on model performance)
            similarities = []
            weights = []
            common_models = set(multi_feat1.keys()) & set(multi_feat2.keys())
            
            # Model weights (can be adjusted based on actual performance)
            model_weights = {'arcface': 0.4, 'adaface': 0.4, 'facenet': 0.2}
            
            for model_name in common_models:
                feat1 = multi_feat1[model_name]
                feat2 = multi_feat2[model_name]
                if feat1 is not None and feat2 is not None:
                    # Use Hungarian algorithm to find best match
                    sim_matrix = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2)
                    sim_matrix_np = sim_matrix.detach().cpu().numpy()
                    cost_matrix = -sim_matrix_np  # Convert to minimization problem
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                    
                    # Extract matched similarity scores
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
            # Default to average method
            return self._compute_ensemble_similarity(multi_feat1, multi_feat2, "average")
    
    def _compute_multimodel_similarity_matrix(self, output_feats, ref_feats, method="average"):
        """Calculate similarity matrix for multi-model features
        
        Args:
            output_feats: output features, may be MultiModelFeatures list or regular tensor
            ref_feats: reference features, may be MultiModelFeatures list or regular tensor  
            method: ensemble method
        """
        # Check if it's multi-model features
        is_multi_output = isinstance(output_feats, list) and len(output_feats) > 0 and isinstance(output_feats[0], MultiModelFeatures)
        is_multi_ref = isinstance(ref_feats, list) and len(ref_feats) > 0 and isinstance(ref_feats[0], MultiModelFeatures)
        
        if is_multi_output or is_multi_ref:
            # Multi-model feature similarity calculation
            if is_multi_output and not is_multi_ref:
                # Output is multi-model, reference is regular features
                similarities = []
                for out_feat in output_feats:
                    # Convert reference features to MultiModelFeatures format
                    ref_multi = MultiModelFeatures({'single': ref_feats})
                    sim = self._compute_ensemble_similarity(out_feat, ref_multi, method)
                    similarities.append([sim])
                return torch.tensor(similarities)
                
            elif not is_multi_output and is_multi_ref:
                # Output is regular features, reference is multi-model
                similarities = []
                # Convert output features to MultiModelFeatures format
                out_multi = MultiModelFeatures({'single': output_feats})
                for ref_feat in ref_feats:
                    sim = self._compute_ensemble_similarity(out_multi, ref_feat, method)
                    similarities.append([sim])
                return torch.tensor(similarities).T
                
            elif is_multi_output and is_multi_ref:
                # Both output and reference are multi-model
                similarities = []
                for out_feat in output_feats:
                    row_sims = []
                    for ref_feat in ref_feats:
                        sim = self._compute_ensemble_similarity(out_feat, ref_feat, method)
                        row_sims.append(sim)
                    similarities.append(row_sims)
                return torch.tensor(similarities)
        
        # Regular feature similarity calculation (original logic)
        # Ensure input is tensor
        if not isinstance(output_feats, torch.Tensor):
            output_feats = torch.tensor(output_feats)
        if not isinstance(ref_feats, torch.Tensor):
            ref_feats = torch.tensor(ref_feats)
        return (output_feats @ ref_feats.T)













    # def dino_detect(self, inp_img: str, inp_cap: str, box_threshold=None, text_threshold=None):
    #     if box_threshold is None:
    #         box_threshold = self.BOX_TRESHOLD
    #     if text_threshold is None:
    #         text_threshold = self.TEXT_TRESHOLD
            
    #     image_source, image = load_image(inp_img)
    #     with torch.no_grad():
    #         boxes, logits, phrases = predict(
    #             model=self.dino_model,
    #             image=image,
    #             caption=inp_cap,
    #             box_threshold=box_threshold,
    #             text_threshold=text_threshold,
    #         )
    #     annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    #     return boxes, logits, phrases, annotated_frame

    def dino_detect(self, inp_img: str, inp_cap: str, box_threshold=None, text_threshold=None):
        if box_threshold is None:
            box_threshold = self.BOX_TRESHOLD
        if text_threshold is None:
            text_threshold = self.TEXT_TRESHOLD
        
        try:
            # Check if file exists
            if not os.path.exists(inp_img):
                print(f"\033[31mError: Image file not found for DINO detection: {inp_img}\033[0m")
                # Return empty detection results
                empty_boxes = torch.empty(0, 4)
                empty_logits = torch.empty(0)
                empty_phrases = []
                return empty_boxes, empty_logits, empty_phrases, None
                
            image_source, image = load_image(inp_img)
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.dino_model,
                    image=image,
                    caption=inp_cap,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            return boxes, logits, phrases, annotated_frame
        except Exception as e:
            print(f"\033[31mError in DINO detection for {inp_img}: {str(e)}\033[0m")
            # Return empty detection results
            empty_boxes = torch.empty(0, 4)
            empty_logits = torch.empty(0)
            empty_phrases = []
            return empty_boxes, empty_logits, empty_phrases, None



    # def crop_img(self, img_src: str, boxes) -> list[Image.Image]:
    #     img_source = Image.open(img_src)
    #     w, h = img_source.size
    #     boxes = (boxes * torch.Tensor([w, h, w, h])).int()
    #     xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    #     cropped_images = []
    #     for box in xyxy:
    #         x_min, y_min, x_max, y_max = box.tolist()
    #         cropped = img_source.crop((x_min, y_min, x_max, y_max))
    #         cropped_images.append(cropped)
    #     return cropped_images

    def crop_img(self, img_src: str, boxes) -> list[Image.Image]:
        try:
            # Check if file exists
            if not os.path.exists(img_src):
                print(f"\033[31mError: Image file not found for cropping: {img_src}\033[0m")
                return []
            
            # Check if boxes is empty
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
                    # Ensure coordinates are within image bounds
                    x_min = max(0, min(x_min, w))
                    y_min = max(0, min(y_min, h))
                    x_max = max(0, min(x_max, w))
                    y_max = max(0, min(y_max, h))
                    
                    if x_max > x_min and y_max > y_min:  # Ensure valid cropping region
                        cropped = img_source.crop((x_min, y_min, x_max, y_max))
                        cropped_images.append(cropped)
                except Exception as e:
                    print(f"\033[33mWarning: Failed to crop box {box}: {str(e)}\033[0m")
                    continue
            return cropped_images
        except Exception as e:
            print(f"\033[31mError in crop_img for {img_src}: {str(e)}\033[0m")
            return []





    def main(self):
        CHARACTER = os.listdir(self.REF_DIR)
        REF_PATH = {char: os.path.join(self.REF_DIR, char) for char in CHARACTER}
        print(f'REF_PATH:{REF_PATH}')
        TEXT_PROMPT = {char: "people" for char in CHARACTER} if not self.PRE_TEXT_PROMPT else self.PRE_TEXT_PROMPT

        if self.ref_mode == 'mid-gen':
            MID_GEN_REF_PATH = {char: None for char in CHARACTER}
            MID_GEN_CHAR_LIST = self.OUTPUT_DIR['chars']

            print(f'MID_GEN_REF_PATH:{MID_GEN_REF_PATH}')
            print(f'MID_GEN_CHAR_LIST:{MID_GEN_CHAR_LIST}')



        # load shots
        shots, Characters = self.load_shots(self.PROMPT_PATH)

        # get ref character features
        ref_clip_feats = {}
        for char in CHARACTER:
            # enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
            enc_name = self._get_encoder_name(Characters[char]["tag"])
            ch_name = Characters[char]['name_ch']
            input_ref_imgs = []

            # ref char from origin-dataset
            if self.ref_mode == 'origin':
                print(f'self.ref_mode:{self.ref_mode}')
                ref_imgs = sorted(glob.glob(os.path.join(REF_PATH[char], '**/*.jpg'), recursive=True))
                # ref_imgs = sorted([f for ext in ['jpg','png'] for f in glob.glob(os.path.join(REF_PATH[char], f'**/*.{ext}'), recursive=True)])
                print(f'ref_imgs:{ref_imgs}')
                '''
                ref_imgs:
                ['data/dataset/ViStory/01/image/Big Brown Rabbit/00.jpg', 
                'data/dataset/ViStory/01/image/Big Brown Rabbit/01.jpg']'''

            # ref char from mid-generated
            elif self.ref_mode == 'mid-gen':
                print(f'self.ref_mode:{self.ref_mode}')

                for img_path in MID_GEN_CHAR_LIST:
                    char_file_name = os.path.basename(img_path).split(".")[0]
                    print(f'char_file_name:{char_file_name}')
                    if char_file_name in MID_GEN_REF_PATH.keys() and char == char_file_name: # 遍历英文名
                        print(f'Find Character {char}({char_file_name}):{img_path}')
                        MID_GEN_REF_PATH[char] = img_path
                        break
                    elif ch_name in char_file_name: # 遍历中文名
                        print(f'Find Character {char}({ch_name}):{img_path}')
                        MID_GEN_REF_PATH[char] = img_path
                        break
                    else:
                        continue

                if MID_GEN_REF_PATH[char] == None:
                    # MID_GEN_REF_PATH[char] = []
                    print(f'[WARN] Missing character: {char}, Assign an empty value [] (available character: {CHARACTER})')
                    continue


                print(f'MID_GEN_REF_PATH:{MID_GEN_REF_PATH}')
                print(f'MID_GEN_CHAR_LIST:{MID_GEN_CHAR_LIST}')


                ref_imgs = sorted(glob.glob(MID_GEN_REF_PATH[char], recursive=True))
                print(f'ref_imgs:{ref_imgs}')
                '''
                ref_imgs:['data/outputs/moki/ViStory_en/01/01_char/Big Brown Rabbit.jpg']'''


            for img in ref_imgs:
                img_file_name = img.split('/')[-1].split('.')[0]
                boxes, logits, _, _ = self.dino_detect(img, TEXT_PROMPT[char])

                # Check if any objects are detected
                if len(logits) == 0:
                    print(f"\033[33mNo objects detected in reference image {img} for {char} with prompt: {TEXT_PROMPT[char]}\033[0m")
                    continue

                _, indices = torch.topk(logits, 1)
                boxes = boxes[indices]
                cropped_imgs = self.crop_img(img, boxes)
                if len(cropped_imgs) != 0:
                    input_ref_imgs.append(cropped_imgs[0]) # assume each ref img cantains only one character
                    # cropped_imgs[0].save(f"{char}-{img_file_name}.png") # NOTE: debug
                else:
                    print(f"\033[33mNo char: {char} found in {img} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")

            # ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
            # if ref_feats is None:
            #     input_ref_imgs = [Image.open(x) for x in ref_imgs]
            #     ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
            # assert ref_feats is not None, f"Cant get ref char: {char}, please check."
            # ref_clip_feats[char] = ref_feats

            # Check if there are valid reference images
            if len(input_ref_imgs) > 0:
                ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
                if ref_feats is None:
                    # Fallback to directly loading original images
                    input_ref_imgs = [Image.open(x) for x in ref_imgs]
                    if len(input_ref_imgs) > 0:
                        ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
            else:
                # No successfully detected reference images, try using original images directly
                print(f"\033[33mNo valid cropped reference images for {char}, trying original images\033[0m")
                input_ref_imgs = [Image.open(x) for x in ref_imgs]
                if len(input_ref_imgs) > 0:
                    ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
                else:
                    ref_feats = None
            
            assert ref_feats is not None, f"Cant get ref char: {char}, please check. No valid reference images found."
            ref_clip_feats[char] = ref_feats


        # get output character features
        # Store results required for cids detection
        results = {"cids": {}}
        # Store completed matching subjects for calculating cids metrics
        char_pil_imgs = {}
        matched_cnt = 0 # Character matching scenario
        omissive_cnt = 0 # Character omission scenario
        superfluous_cnt = 0 # Character surplus scenario

        for shot in shots:
            shot_results = {}
            shot_id = int(shot["index"]) - 1
            print(f"self.OUTPUT_DIR['shots'][:3]:{self.OUTPUT_DIR['shots'][:3]}")

            # Check if shot_id is out of bounds (added to handle missing subsequent shots like seed story)
            if shot_id >= len(self.OUTPUT_DIR['shots']):
                print(f"Warning: shot_id {shot_id} is out of bounds. Skipping...")
                continue  # Skip current shot
            
            # target_img_path = os.path.join(self.OUTPUT_DIR['shots'], f"{shot_id}_0.png") 
            target_img_path = self.OUTPUT_DIR['shots'][shot_id]
            print(f'Current shot generated image: {target_img_path}')

            # Check if image file exists
            if not os.path.exists(target_img_path):
                print(f"\033[31mError: Image file not found: {target_img_path}. Skipping shot {shot_id}.\033[0m")
                # Mark all characters in this shot as missing
                for char in shot['Characters Appearing']['en']:
                    shot_results.update({ char: { 
                        "box": "null",
                        "cross_sim": 0.0
                    } })
                omissive_cnt += 1
                results.update({f"shot-{shot_id}": shot_results})
                continue

            is_omissive_shot = False

            # Retrieve possible subjects in generated images through character keywords
            for char in shot['Characters Appearing']['en']:
                # enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
                enc_name = self._get_encoder_name(Characters[char]["tag"])
                if char not in char_pil_imgs.keys():
                    char_pil_imgs.update({char: []})
                boxes, logits, phrases, annotated_frame = self.dino_detect(target_img_path, TEXT_PROMPT[char])
                
                # Check if there are detection results
                if len(logits) == 0:
                    print(f"\033[33mNo objects detected for {char} in {target_img_path}\033[0m")
                    shot_results.update({ char: { 
                        "box": "null",
                        "cross_sim": 0.0
                    } })
                    continue
                
                # Select the most relevant subjects not exceeding the number of characters required in the current scene
                _, indices = torch.topk(logits, min(len(logits), len(shot['Characters Appearing']['en'])))
                boxes = boxes[indices]
                # cv2.imwrite(f"{shot_id}-{char}.png", annotated_frame) # NOTE: debug
                # Crop possible subjects corresponding to keywords
                cropped_imgs = self.crop_img(target_img_path, boxes)
                if len(cropped_imgs) != 0:
                    output_feats = self.get_char_feat(cropped_imgs, encoder_name=enc_name)
                    if output_feats is None:
                        print(f"\033[33mNo char: {char} found in {target_img_path} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
                        # When character detection fails in a scene, leave box data empty
                        shot_results.update({ char: { 
                            "box": "null",
                            "cross_sim": 0.0
                        } })
                        continue

                    cosine_sims = {}
                    # Pre-calculate similarity of each subject to each character reference image (take maximum of multi-image cosine similarity coefficients)
                    # for _char in shot['Characters Appearing']['en']:
                    #     if _char not in ref_clip_feats.keys():
                    #         continue
                    #     cosine_sims.update({_char: (output_feats.cuda() @ ref_clip_feats[_char].cuda().T).max(dim=1).values.cpu()})
                    
                    # Pre-calculate similarity of each subject to each character reference image (take maximum of multi-image cosine similarity coefficients)
                    for _char in shot['Characters Appearing']['en']:
                        # Use new multi-model similarity calculation method
                        similarity_matrix = self._compute_multimodel_similarity_matrix(
                            output_feats.cuda() if not isinstance(output_feats, list) else output_feats,
                            ref_clip_feats[_char].cuda() if not isinstance(ref_clip_feats[_char], list) else ref_clip_feats[_char],
                            method=self.ensemble_method  # Use average method for multi-model fusion
                        )
                        cosine_sims.update({_char: similarity_matrix.max(dim=1).values.cpu()})


                    print(f'cosine_sims.keys():{cosine_sims.keys()}')
                    print(f'char:{char}')
                    if char not in cosine_sims.keys():
                        continue

                    sims_bak = cosine_sims[char].tolist()
                    available_indices = list(range(len(sims_bak)))  # Track available original indices
                    matched_flag = False

                    # Starting from the most similar subject, check one by one if any other character has higher similarity to it;
                    # If yes, exclude this subject and check the next most similar subject; if not, match this subject to this character
                    while(len(sims_bak) > 0):

                        # _id = torch.argmax(torch.Tensor(sims_bak)).item()
                        # for _char in shot['Characters Appearing']['en']:
                        #     if _char not in cosine_sims.keys():
                        #         continue
                        #     if _char != char:
                        #         if cosine_sims[_char][_id] > cosine_sims[char][_id]:
                        #             break
                        # else:
                        #     matched_flag = True
                        #     boxes_id = cosine_sims[char].tolist().index(sims_bak[_id])
                        #     break
                        # sims_bak.pop(_id)


                        local_id = torch.argmax(torch.Tensor(sims_bak)).item()  # Local index in sims_bak
                        original_id = available_indices[local_id]  # Corresponding index in original tensor
                        
                        # Check if any other character has higher similarity to this box
                        conflict_found = False
                        for _char in shot['Characters Appearing']['en']:
                            if _char != char:
                                # Ensure both character similarity tensors have sufficient elements
                                if (original_id < len(cosine_sims[_char]) and 
                                    original_id < len(cosine_sims[char]) and 
                                    cosine_sims[_char][original_id] > cosine_sims[char][original_id]):
                                    conflict_found = True
                                    break
                        
                        if not conflict_found:
                            matched_flag = True
                            boxes_id = original_id
                            break
                        
                        # Remove conflicting candidate
                        sims_bak.pop(local_id)
                        available_indices.pop(local_id)


                    # if matched_flag:
                    #     boxes_to_write = [round(x, 3) for x in boxes[boxes_id].tolist()]
                    #     shot_results.update({ char: { "box": boxes_to_write} })
                    #     char_pil_imgs[char].append(cropped_imgs[boxes_id])
                    #     cropped_imgs[boxes_id].save(f"{self.mid_result_dir}/shot{shot_id:02d}-{char}.png") # NOTE:debug

                    # else:
                    #     # When character detection fails in a scene, leave box data empty
                    #     shot_results.update({ char: { "box": "null" } })


                    if matched_flag:
                        boxes_to_write = [round(x, 3) for x in boxes[boxes_id].tolist()]
                        # Calculate cross_sim for this character in current shot
                        try:
                            matched_output_feat = output_feats[boxes_id:boxes_id+1] if not isinstance(output_feats, list) else [output_feats[boxes_id]]
                            cross_sim_matrix = self._compute_multimodel_similarity_matrix(
                                matched_output_feat,
                                ref_clip_feats[char],
                                method=self.ensemble_method
                            )
                            if cross_sim_matrix.numel() > 0:  # Check if tensor is empty
                                shot_cross_sim = round(cross_sim_matrix.mean().item(), 4)
                            else:
                                shot_cross_sim = 0.0
                        except Exception as e:
                            print(f"\033[33mWarning: Failed to compute cross_sim for {char} in shot {shot_id}: {str(e)}\033[0m")
                            shot_cross_sim = 0.0
                        
                        shot_results.update({ char: { 
                            "box": boxes_to_write,
                            "cross_sim": shot_cross_sim
                        }})
                        char_pil_imgs[char].append(cropped_imgs[boxes_id])
                        cropped_imgs[boxes_id].save(f"{self.mid_result_dir}/shot{shot_id:02d}-{char}.png") # NOTE:debug

                    else:
                        # When character detection fails in a scene, leave box data empty
                        shot_results.update({ char: { 
                            "box": "null",
                            "cross_sim": 0.0
                        } })


                else:
                    # print(f"\033[33mNo char: {char} found in {target_img_path} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
                    # # When character detection fails in a scene, leave box data empty
                    # shot_results.update({ char: { "box": "null" } })

                    print(f"\033[33mNo char: {char} found in {target_img_path} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
                    # When character detection fails in a scene, leave box data empty
                    shot_results.update({ char: { 
                        "box": "null",
                        "cross_sim": 0.0
                    } })

                    
            # Detect missing characters by checking if bounding box exists
            if any([x["box"] == "null" for x in shot_results.values()]):
                omissive_cnt += 1
                is_omissive_shot = True

            # For scenes that originally don't contain characters, detect if there are surplus characters
            is_superfluous_shot = False
            if len(shot['Characters Appearing']['en']) == 0:
                for char in CHARACTER:
                    # enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
                    enc_name = self._get_encoder_name(Characters[char]["tag"])
                    boxes, logits, phrases, annotated_frame = self.dino_detect(target_img_path, TEXT_PROMPT[char])

                    # _, indices = torch.topk(logits, min(len(logits), 5))
                    # boxes = boxes[indices]

                    # Check if there are detection results
                    if len(logits) == 0:
                        continue  # No objects detected, continue to next character
                    
                    k = min(len(logits), 5)
                    if k > 0:
                        _, indices = torch.topk(logits, k)
                        boxes = boxes[indices]
                    else:
                        continue  # No valid boxes


                    # Crop possible subjects corresponding to keywords
                    cropped_imgs = self.crop_img(target_img_path, boxes)
                    if len(cropped_imgs) != 0:
                        output_feats = self.get_char_feat(cropped_imgs, encoder_name=enc_name)
                        if output_feats is not None:
                            # When similarity is greater than 0.8, determine that this scene contains a specific character

                            # cosine_sims = {}
                            # for _char in shot['Characters Appearing']['en']:
                            #     if _char not in ref_clip_feats.keys():
                            #         continue
                            #     cosine_sims.update({_char: (output_feats @ ref_clip_feats[_char].T).max(dim=1).values.cpu()})
                            # if any([x > 0.8 for x in cosine_sims.values()]):
                            #     is_superfluous_shot = True


                            # Use new multi-model similarity calculation method
                            similarity_matrix = self._compute_multimodel_similarity_matrix(
                                output_feats,
                                ref_clip_feats[char],
                                method=self.ensemble_method  # Use average method for multi-model fusion
                            )
                            max_similarity = similarity_matrix.max().item()
                            if max_similarity > 0.8:
                                is_superfluous_shot = True
                                break  # Exit loop once surplus character is found

                if is_superfluous_shot:
                    superfluous_cnt += 1

            # If the scene has neither omission nor surplus, it belongs to the matching scenario
            if is_omissive_shot is not True and is_superfluous_shot is not True:
                matched_cnt += 1

            results.update({f"shot-{shot_id}": shot_results})

        num_shots = len(shots)
        results.update(
            {
                "matched-shots": f"{matched_cnt} / {num_shots}",
                "omissive-shots": f"{omissive_cnt} / {num_shots}",
                "superfluous-shots": f"{superfluous_cnt} / {num_shots}",
            }
        )

        # You can uncomment the following code to manually check if characters are correctly identified
        # for char in char_pil_imgs.keys(): # NOTE: for debug
        #     for idx, img in enumerate(char_pil_imgs[char]):
        #         img.save(f"{char}-{idx}.png")

        # Calculate copy-paste metrics
        copy_paste_cnt = 0
        shot_copy_paste_score = 0.0
        # Calculate cids cross and self metrics separately
        for char in CHARACTER:
            # enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
            enc_name = self._get_encoder_name(Characters[char]["tag"])
            # if len(char_pil_imgs[char]) > 0:
            if char in char_pil_imgs and len(char_pil_imgs[char]) > 0:
                char_feats = self.get_char_feat(char_pil_imgs[char], encoder_name=enc_name)
                if char_feats is None:
                    # When no character is detected throughout, score is 0.0
                    results["cids"].update({char: {
                        "cross": 0.0,
                        "self": 0.0,
                    }})
                    continue
                
                # if char not in ref_clip_feats.keys():
                #     continue

                # cross_sim = (char_feats @ ref_clip_feats[char].T).mean().item()
                # self_sim = (char_feats @ char_feats.T).mean().item()
                # results["cids"].update({char: {
                #     "cross": round(cross_sim, 4),
                #     "self": round(self_sim, 4)
                # }})

                # # copy-paste
                # if ref_clip_feats[char].shape[0] > 1:
                #     copy_paste_cnt += 1
                #     cross_sims = char_feats @ ref_clip_feats[char].T
                #     copy_paste_score = (cross_sims[:, 0].unsqueeze(-1) - cross_sims[:, 1:]).mean().item()
                #     shot_copy_paste_score += copy_paste_score



                # Use new multi-model similarity calculation method to calculate cross similarity
                cross_similarity_matrix = self._compute_multimodel_similarity_matrix(
                    char_feats,
                    ref_clip_feats[char],
                    method=self.ensemble_method  # Use average method for multi-model fusion
                )
                cross_sim = cross_similarity_matrix.mean().item()
                
                # Use new multi-model similarity calculation method to calculate self similarity
                self_similarity_matrix = self._compute_multimodel_similarity_matrix(
                    char_feats,
                    char_feats,
                    method=self.ensemble_method  # Use average method for multi-model fusion
                )
                if self_similarity_matrix.shape[0] > 1:
                    # Get upper triangular matrix, excluding diagonal
                    indices = torch.triu_indices(self_similarity_matrix.shape[0], self_similarity_matrix.shape[1], offset=1)
                    self_sim = self_similarity_matrix[indices[0], indices[1]].mean().item()
                else:
                    # If character only appears once, self-sim cannot be calculated, can be defined as 1.0 or 0.0, here set to 1.0
                    self_sim = 1.0
                results["cids"].update({char: {
                    "cross": round(cross_sim, 4),
                    "self": round(self_sim, 4)
                }})
                # copy-paste
                # Need to check the number of reference features, special handling required for multi-model features
                ref_count = len(ref_clip_feats[char]) if isinstance(ref_clip_feats[char], list) else ref_clip_feats[char].shape[0]
                if ref_count > 1:
                    copy_paste_cnt += 1
                    # Use new multi-model similarity calculation method
                    cross_sims = self._compute_multimodel_similarity_matrix(
                        char_feats,
                        ref_clip_feats[char],
                        method=self.ensemble_method
                    )
                    # Ensure cross_sims has enough columns for comparison
                    if cross_sims.shape[1] > 1:
                        copy_paste_score = (cross_sims[:, 0].unsqueeze(-1) - cross_sims[:, 1:]).mean().item()
                        shot_copy_paste_score += copy_paste_score
                    else:
                        # If only one column, skip copy-paste calculation
                        copy_paste_cnt -= 1  # Subtract back the count just increased


            else:
                # When no character is detected throughout, score is 0.0
                results["cids"].update({char: {
                    "cross": 0.0,
                    "self": 0.0,
                }})
        if copy_paste_cnt > 0:
            shot_copy_paste_score /= copy_paste_cnt
            results.update({"copy-paste-score": shot_copy_paste_score,})
        else:
            results.update({"copy-paste-score": "null"})
        # print(results)
        return results



if __name__ == "__main__":
    # Example usage:
    # pre_fix = "/path/to/your/data"
    # output_dir = "/path/to/your/output"
    # cids = CIDS(pre_fix, output_dir)
    # results = cids.get_cids_score()
    # print(results)
    pass