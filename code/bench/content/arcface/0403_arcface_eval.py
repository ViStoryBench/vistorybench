from metrics.arcface_metric import ArcFace_Metrics
from PIL import Image
import os
from metrics.clip_metric import CLIP_Metric
import json
from metrics.aesthetic_metric import Aesthetic_Metric
from metrics.anatomy_metric import Anatomy_Metric

from metrics.fid_metric import FID_Metric


if __name__ == '__main__':
    metrics = ArcFace_Metrics()
    clip_metric = CLIP_Metric()
    aesthetic_metrics = Aesthetic_Metric()
    anatomical_metrics = Anatomy_Metric()
    fid_metrics = FID_Metric()
    target_dir = "/data/MIBM/UniPortrait/output/demo"
    list_dir = os.listdir(target_dir)

    output_dir = "./output/uniportrait"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Evaluating...")
    sim_ori_list = []
    sim_ref_list = []
    sim_clip_i_list = []
    sim_clip_t_list = []
    sim_unmatched_list = []
    sim_unmatched_ref_list = []
    sim_cluster_list = []
    aes_list = []
    total_count = 0
    bad_count = 0
    normal_count = 0

    anatomy_bad_count = 0

    # collect all the images for FID
    original_images = []
    generated_images = []



    for dir in list_dir:
        total_count += 1
        # caption = json.load(open(os.path.join(target_dir, dir, "meta.json")))["caption_en"]
        meta = json.load(open(os.path.join(target_dir, dir, "meta.json")))
        caption = meta["caption_en"]
        names = meta["name"]
        ori_img = Image.open(os.path.join(target_dir, dir, "ori.jpg"))
        output_img = Image.open(os.path.join(target_dir, dir, "output_0.jpg"))
        ref_1 = Image.open(os.path.join(target_dir, dir, "ref_1.jpg"))
        ref_2 = Image.open(os.path.join(target_dir, dir, "ref_2.jpg"))
        refs = [ref_1, ref_2]
        # sim_ori = sim = metrics.compare_mutli_person_on_2_images(output_img, ori_img)
        # sim_ref = metrics.compare_mutli_person_on_2_images_with_ref(output_img, refs)

        _, sim_ori, non_matched_mean_ori = metrics.compare_faces_with_confusion_matrix(output_img, ori_img) # 和单张图片上的人物对比
        _, sim_ref, non_matched_mean_ref = metrics.compare_faces_with_confusion_matrix_with_ref(output_img, refs) # 和多张图片上的人物对比
        _, sim_cluster, non_matched_mean_cluster = metrics.compare_faces_with_confusion_matrix_with_ref_names(output_img, names)
        # compare clip-i between ori and output
        clip_i = clip_metric.compute_clip_i(ori_img, output_img)
        clip_t = clip_metric.compute_clip_t(output_img, caption)
        aes_score = aesthetic_metrics(output_img)
        if sim_ori is None or sim_ref is None:
            bad_count += 1

            continue

        

        anatomical_score = anatomical_metrics(output_img, return_scores=True)[0]

        print("Base Name:", dir)
        print(f"Similarity with ori: {sim_ori}, Similarity with ref: {sim_ref}")
        print(f"Avg Sim Ori: {sim_ori.mean()}, Avg Sim Ref: {sim_ref.mean()}, Diff {sim_ori.mean() - sim_ref.mean()}, Non Matched Ori: {non_matched_mean_ori}, Non Matched Ref: {non_matched_mean_ref}")
        print("Avg Sim Cluster:", sim_cluster.mean())
        print(f"Aesthetic Score: {aes_score}")

        # if anatomical_bool:
        #     print("Anatomy is good")
        # else:
        #     anatomy_bad_count += 1
        #     print("Anatomy is bad")
        print(f"Anatomy Score: {anatomical_score}")

        print(f"Clip I: {clip_i}, Clip T: {clip_t}")
        sim_ori_list.append(sim_ori.mean())
        sim_ref_list.append(sim_ref.mean())
        sim_unmatched_list.append(non_matched_mean_ori)
        sim_unmatched_ref_list.append(non_matched_mean_ref)
        sim_clip_i_list.append(clip_i)
        sim_clip_t_list.append(clip_t)
        sim_cluster_list.append(sim_cluster.mean())
        aes_list.append(aes_score)
        normal_count += 1

        # Collect images for FID
        original_images.append(ori_img)
        generated_images.append(output_img)

        ##### debug: save the image, and a json containing its Anatomy score on top
        # if not os.path.exists(os.path.join(output_dir, dir)):
        #     os.makedirs(os.path.join(output_dir, dir))
        output_img.save(os.path.join(output_dir, meta["id"] + ".jpg"))
        with open(os.path.join(output_dir,  meta["id"] + ".json"), "w") as f:
            json.dump({
                "anatomy_score": float(anatomical_score) if isinstance(anatomical_score, (float, int)) or hasattr(anatomical_score, "item") else anatomical_score,
                "caption": caption,
                "sim_ori": float(sim_ori.mean()),
                "sim_ref": float(sim_ref.mean()),
                "sim_cluster": float(sim_cluster.mean()),
                "clip_i": float(clip_i),
                "clip_t": float(clip_t),
                "aes_score": float(aes_score)
            }, f, indent=4)

    # Calculate FID score
    print("Calculating FID score...")
    fid_score = fid_metrics.calculate_fid(original_images, generated_images)
    print(f"FID Score: {fid_score}")


    print(f"Done evaluation, Avg Sim Ori: {sum(sim_ori_list) / len(sim_ori_list)}, Avg Sim Ref: {sum(sim_ref_list) / len(sim_ref_list)}, Avg Diff: {(sum(sim_ori_list) / len(sim_ori_list)) - (sum(sim_ref_list) / len(sim_ref_list))}")
    print(f"Avg Sim Cluster: {sum(sim_cluster_list) / len(sim_cluster_list)}")
    print("Avg Diff(Ori - Cluster):", (sum(sim_ori_list) / len(sim_ori_list)) - (sum(sim_cluster_list) / len(sim_cluster_list)))
    print(f"Avg Clip I: {sum(sim_clip_i_list) / len(sim_clip_i_list)}")
    print(f"Avg Clip T: {sum(sim_clip_t_list) / len(sim_clip_t_list)}")
    print(f"Avg Non Matched Ori: {sum(sim_unmatched_list) / len(sim_unmatched_list)}")
    print(f"Avg Non Matched Ref: {sum(sim_unmatched_ref_list) / len(sim_unmatched_ref_list)}")
    print(f"Avg Aesthetic Score: {sum(aes_list) / len(aes_list)}")
    print(f"In total {total_count} images, {bad_count} bad samples")