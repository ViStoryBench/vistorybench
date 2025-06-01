import insightface
import cv2
import numpy as np
import os
# use torch cos similarity
from torch.nn import functional as F
import scipy.optimize
import torch

from PIL import Image


class ArcFace_Metrics():
    def __init__(self, model_path = "./", ref_dir =  "/mnt/xuhengyuan/data/2person/v1/0105_ref_cluster/", device='cuda'):
        self.device = device
        self.model = insightface.app.FaceAnalysis(name = "antelopev2", root=model_path, providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=0, det_thresh=0.45)
        self.cos = F.cosine_similarity
        self.ref_dir = ref_dir

    def get_embeddings(self, img):
        if not isinstance(img, np.ndarray):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        res = self.model.get(img)
        if len(res) == 0:
            return None
        # return res.embedding
        # return torch.stack([r.embedding for r in res])
        # [r.embedding for r in res] is list of np.array, need to convert to torch.tensor
        return torch.stack([torch.from_numpy(r.embedding) for r in res])
    def get_multi_img_embeddings(self, imgs):
        results = []
        for img in imgs:
            if not isinstance(img, np.ndarray):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            res = self.model.get(img)
            if len(res) == 0:
                pass
            else:
                results.append(res)
        # convert them to torch.tensor
        return torch.stack([torch.from_numpy(r[0].embedding) for r in results]) 
        

        
    
    def compare_mutli_person_on_2_images(self, img1, img2):
        
        embs_1 = self.get_embeddings(img1)
        embs_2 = self.get_embeddings(img2)

        if embs_1.shape[0] != embs_2.shape[0]:
            # raise ValueError('The number of faces in two images are not the same')
            print("Warning: The number of faces in two images are not the same")
            return None
        
        sim_matrix =  self.cos(embs_1.unsqueeze(1), embs_2.unsqueeze(0), dim=2)

        # torch.zeros((embs_1.shape[0], embs_2.shape[0]), device=self.device)
        # for i in range(embs_1.shape[0]):
        #     for j in range(embs_2.shape[0]):
        #         sim_matrix[i,j] = self.cos(embs_1[i].unsqueeze(0), embs_2[j].unsqueeze(0))
        
        # print(sim_matrix)

        # print("sim between 1 and 1", self.cos(embs_1[0].unsqueeze(0), embs_2[0].unsqueeze(0)))
        # print("sim between 2 and 2", self.cos(embs_1[1].unsqueeze(0), embs_2[1].unsqueeze(0)))  
        # print("sim between 1 and 2", self.cos(embs_1[0].unsqueeze(0), embs_2[1].unsqueeze(0)))
        # print("sim between 2 and 1", self.cos(embs_1[1].unsqueeze(0), embs_2[0].unsqueeze(0)))

        # 使用匈牙利算法找到最佳匹配
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        cost_matrix = -sim_matrix_np  # 转换为最小化问题
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

        # 提取对应的相似度分数
        row_ind = torch.from_numpy(row_ind).to(sim_matrix.device)
        col_ind = torch.from_numpy(col_ind).to(sim_matrix.device)
        selected_scores = sim_matrix[row_ind, col_ind]

        return selected_scores
    
    def compare_mutli_person_on_2_images_with_ref(self, img, refs):
        embs_1 = self.get_embeddings(img)
        embs_2 = self.get_multi_img_embeddings(refs)
        # print("embs_1.shape: ", embs_1.shape)
        # print("embs_2.shape: ", embs_2.shape)
        if embs_1.shape[0] != embs_2.shape[0]:
            # raise ValueError('The number of faces in two images are not the same')
            print("Warning: The number of faces in two images are not the same")
            return None

        sim_matrix =  self.cos(embs_1.unsqueeze(1), embs_2.unsqueeze(0), dim=2)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        cost_matrix = -sim_matrix_np  # 转换为最小化问题
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

        # 提取对应的相似度分数
        row_ind = torch.from_numpy(row_ind).to(sim_matrix.device)
        col_ind = torch.from_numpy(col_ind).to(sim_matrix.device)
        selected_scores = sim_matrix[row_ind, col_ind]

        return selected_scores
    def compare_faces_with_confusion_matrix(self, img1, img2):
        """
        Compare faces between two images and return:
        1. The full similarity matrix (confusion matrix)
        2. The list of scores for matched face pairs (diagonal elements after optimal assignment)
        3. The mean score of non-matched face pairs (non-diagonal elements)
        """
        # Get embeddings for faces in both images
        embs_1 = self.get_embeddings(img1)
        embs_2 = self.get_embeddings(img2)
        
        if embs_1 is None or embs_2 is None:
            print("Warning: No faces detected in one or both images")
            return None, None, None
        
        # Calculate similarity matrix (cosine similarity between all pairs of embeddings)
        sim_matrix = self.cos(embs_1.unsqueeze(1), embs_2.unsqueeze(0), dim=2)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        
        # Use Hungarian algorithm to find optimal matching
        cost_matrix = -sim_matrix_np  # Convert to minimization problem (maximize similarity)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        
        # Extract matched scores (diagonal elements after optimal assignment)
        matched_scores = sim_matrix_np[row_ind, col_ind]
        
        # Create mask to identify non-matched pairs
        mask = np.ones_like(sim_matrix_np, dtype=bool)
        for r, c in zip(row_ind, col_ind):
            mask[r, c] = False
        
        # Calculate mean of non-matched scores (non-diagonal elements)
        non_matched_mean = np.mean(sim_matrix_np[mask]) if np.any(mask) else None
        
        return sim_matrix_np, matched_scores, non_matched_mean
    
    def compare_faces_with_confusion_matrix_with_ref(self, img, refs):
        """
        Compare faces between an image and multiple reference images and return:
        1. The full similarity matrix
        2. The list of scores for matched face pairs
        3. The mean score of non-matched face pairs
        
        Each reference image should contain one face.
        """
        embs_1 = self.get_embeddings(img)
        embs_2 = self.get_multi_img_embeddings(refs)
        
        if embs_1 is None or embs_2 is None:
            print("Warning: No faces detected in image or references")
            return None, None, None
        
        # Calculate similarity matrix
        sim_matrix = self.cos(embs_1.unsqueeze(1), embs_2.unsqueeze(0), dim=2)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        
        # Use Hungarian algorithm to find optimal matching
        cost_matrix = -sim_matrix_np  # Convert to minimization problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        
        # Extract matched scores
        matched_scores = sim_matrix_np[row_ind, col_ind]
        
        # Create mask to identify non-matched pairs
        mask = np.ones_like(sim_matrix_np, dtype=bool)
        for r, c in zip(row_ind, col_ind):
            mask[r, c] = False
        
        # Calculate mean of non-matched scores
        non_matched_mean = np.mean(sim_matrix_np[mask]) if np.any(mask) else None
        
        return sim_matrix_np, matched_scores, non_matched_mean
    
    def compare_faces_with_confusion_matrix_with_ref_embeddings(self, img, ref_embeddings):
        """
        Compare faces between an image and multiple reference embeddings and return:
        1. The full similarity matrix
        2. The list of scores for matched face pairs
        3. The mean score of non-matched face pairs
        
        Each reference embedding should correspond to one face.
        """
        embs_1 = self.get_embeddings(img)
        
        if embs_1 is None or ref_embeddings is None:
            print("Warning: No faces detected in image or references")
            return None, None, None
        
        # Calculate similarity matrix
        sim_matrix = self.cos(embs_1.unsqueeze(1), ref_embeddings.unsqueeze(0), dim=2)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        
        # Use Hungarian algorithm to find optimal matching
        cost_matrix = -sim_matrix_np
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        # Extract matched scores
        matched_scores = sim_matrix_np[row_ind, col_ind]
        # Create mask to identify non-matched pairs
        mask = np.ones_like(sim_matrix_np, dtype=bool)
        for r, c in zip(row_ind, col_ind):
            mask[r, c] = False
        # Calculate mean of non-matched scores
        non_matched_mean = np.mean(sim_matrix_np[mask]) if np.any(mask) else None
        return sim_matrix_np, matched_scores, non_matched_mean
    
    def compare_faces_with_confusion_matrix_with_ref_names(self, img, names):
        ref_embeddings = self.retrive_cluster_center(names)
        return self.compare_faces_with_confusion_matrix_with_ref_embeddings(img, ref_embeddings)
    
    def retrive_cluster_center(self, names):
        num_name = len(names)
        cluster_center = []
        for i in range(num_name):
            name = names[i]
            npy = np.load(os.path.join(self.ref_dir, name + ".npy"))

            # # load "embeddings" or "embedding" key
            # if "embeddings" in npy:
            #     embeddings = npy["embeddings"]
            # elif "embedding" in npy:
            #     embeddings = npy["embedding"]
            # else:
            #     raise ValueError("No embeddings or embedding key found in the .npy file")
            cluster_center.append(npy)
        cluster_center = np.array(cluster_center)
        return torch.from_numpy(cluster_center)




if __name__ == '__main__':
    metrics = ArcFace_Metrics()
    ori_img = Image.open("/data/MIBM/UniPortrait/output/demo/4e214a20c8f30f40c9b832feb35277f5b2d28fb9/ori.jpg")
    output_img = Image.open("/data/MIBM/UniPortrait/output/demo/4e214a20c8f30f40c9b832feb35277f5b2d28fb9/output_0.jpg")
    sim = metrics.compare_mutli_person_on_2_images(output_img, ori_img)
    # test confuision matrix
    sim_matrix, matched_scores, non_matched_mean = metrics.compare_faces_with_confusion_matrix(ori_img, output_img)
    print("matched scores: ", matched_scores)
    print("non matched mean: ", non_matched_mean)
    print("sim between ori and output: ", sim)
    ref_1 = Image.open("/data/MIBM/UniPortrait/output/demo/4e214a20c8f30f40c9b832feb35277f5b2d28fb9/ref_1.jpg")
    ref_2 = Image.open("/data/MIBM/UniPortrait/output/demo/4e214a20c8f30f40c9b832feb35277f5b2d28fb9/ref_2.jpg")

    refs = [ref_1, ref_2]
    sim = metrics.compare_mutli_person_on_2_images_with_ref(output_img, refs)
    # test confusion matrix with refs
    sim_matrix, matched_scores, non_matched_mean = metrics.compare_faces_with_confusion_matrix_with_ref(output_img, refs)
    print("matched scores: ", matched_scores)
    print("non matched mean: ", non_matched_mean)
    print("sim between output and refs: ", sim)

    

    