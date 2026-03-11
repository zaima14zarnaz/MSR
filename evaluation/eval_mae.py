import sys
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision
from torchvision.utils import draw_bounding_boxes
import numpy as np
from torch.utils.data import random_split, ConcatDataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from scipy.stats import spearmanr
from datetime import datetime
import cv2
import json
from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation

# from bsd.bsd import MultiScaleSaliency
# from bsd.roi_feats_fusion_adaptive import MultiScaleSaliencyMAFormer
from seg_utility import mask_to_coco_polygon
from seg_utility import polygon_to_mask
from seg_utility import mask_iou
from seg_utility import batch_iou_masks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_process.seg_model_dataloader import SaliencyDataset
from pre_process.seg_model_collate import variable_collate_fn



device = "cuda:0" # torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
splits = {"train": "train", "val": "val", "test": "test"}
dataset_name = "IRSR_ASSR"
# dataset_name = "ASSR"
# dataset_name = "SIFR"
# dataset_dir = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR"
dataset_dir = "/data/research/zaima/dataset/Dataset/ASSR" 
# dataset_dir = "/data/research/zaima/dataset/Dataset/SIFR_dataset"
images_store = "images"
ranks_order_store = "rank_order"


val_dataset = SaliencyDataset(
    image_dir=os.path.join(dataset_dir, images_store, splits["test"]),
    rank_dir=os.path.join(dataset_dir, ranks_order_store, splits["test"]),
    obj_seg_json=os.path.join(dataset_dir, f"obj_seg_data_{splits['test']}.json"),
    img_size=(480, 640),
    descriptions_csv = None # os.path.join(dataset_dir, "test.csv")
)

val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=variable_collate_fn
)
    
def tensor_iou(m1, m2):
    """m1: (H,W), m2: (H,W) binary float masks"""
    inter = (m1 * m2).sum()
    union = m1.sum() + m2.sum() - inter
    if union <= 0:
        return 0.0
    return (inter / union).item()


# run_save_dir = os.path.join(model_save_path, f"run_{run_no}")
# os.makedirs(run_save_dir, exist_ok=True)

# ------------------------- EVALUATION -------------------------
val_loss_sum = 0.0
val_mae_sum  = 0.0

save_dir = "/data/research/zaima/dataset/Dataset/pred_masks"
save_dir = os.path.join(save_dir, dataset_name)
os.makedirs(save_dir, exist_ok=True)

save_pred_masks = False
saved_data_path = "sr_ranks.json"
with open(saved_data_path, "r") as f:
    gt_data = json.load(f)

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch.nn.functional as F

class Mask2FormerConfig:
    backbone_resnet50 = "resnet50"
    backbone_resnet101 = "resnet101"
    backbone_swinb = "swinb"
    backbone_swinl = "swinl"
    model_name_resnet50 = "ClownRat/mask2former-resnet-50-coco-instance"
    model_name_resnet101 = "ClownRat/mask2former-resnet-101-coco-instance"
    model_name_swinb = "facebook/maskformer-swin-base-coco"
    model_name_swinl = "facebook/maskformer-swin-large-coco"
    res50_conf_thres = 0.0
    res101_conf_thres = None
    swinb_conf_thres = 0.5
    swinl_conf_thres = 0.5

    
backbone = Mask2FormerConfig.backbone_swinl
if backbone == Mask2FormerConfig.backbone_resnet50:
    processor = AutoImageProcessor.from_pretrained(Mask2FormerConfig.model_name_resnet50)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(Mask2FormerConfig.model_name_resnet50).to(device)
    conf_thesh = Mask2FormerConfig.res50_conf_thres
if backbone == Mask2FormerConfig.backbone_resnet101:
    processor = AutoImageProcessor.from_pretrained(Mask2FormerConfig.model_name_resnet101)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(Mask2FormerConfig.model_name_resnet101).to(device)
    conf_thesh = Mask2FormerConfig.res101_conf_thres
if backbone == Mask2FormerConfig.backbone_swinb:
    processor = AutoImageProcessor.from_pretrained(Mask2FormerConfig.model_name_swinb)
    model = MaskFormerForInstanceSegmentation.from_pretrained(Mask2FormerConfig.model_name_swinb).to(device)
    conf_thesh = Mask2FormerConfig.swinb_conf_thres
if backbone == Mask2FormerConfig.backbone_swinl:
    processor = AutoImageProcessor.from_pretrained(Mask2FormerConfig.model_name_swinl)
    model = MaskFormerForInstanceSegmentation.from_pretrained(Mask2FormerConfig.model_name_swinl).to(device)
    conf_thesh = Mask2FormerConfig.swinl_conf_thres

model.eval()

val_loss_total = 0.0
total_val_mae = 0.0
num_val_mae_samples = 0
height, width = 480, 640

# Precompute GT masks from polygons ONCE
if save_pred_masks:
    precomputed_gt_masks = {}  # image_id -> (N_obj, H, W) torch.FloatTensor on GPU
    for image_id, entry in tqdm(gt_data.items(), desc="GT Mask Precompute", unit="image"):
        obj_masks = []
        object_list = entry["object_data"]
        for obj in object_list:
            polys = obj["segmentation"]
            if not polys:
                continue
            # Convert polygon -> mask
            mask_np = polygon_to_mask(polys, height, width)
            obj_masks.append(torch.from_numpy(mask_np).float())
        # Stack / store results
        if obj_masks:
            precomputed_gt_masks[image_id] = torch.stack(obj_masks, dim=0).to(device)
        else:
            precomputed_gt_masks[image_id] = torch.zeros((0, height, width), dtype=torch.float32, device=device)


total_objs = 0
no_seg_mask_obj = 0
pbar_val = tqdm(
        val_loader,
        desc=f"Evaluating",
        dynamic_ncols=True,
        unit="batch",
        leave=False,
        disable=not sys.stdout.isatty()
        )
img_area = 640 * 480
with torch.inference_mode():
    for img_ids, pil_images, gts, rank_lists, gt_classes_batch, inst_masks, rois, phrases in pbar_val:

        B = len(pil_images)

        # get original image sizes BEFORE model inference
        target_sizes = [(img.height, img.width) for img in pil_images]

        # forward pass
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        outputs = model(**inputs)

        if conf_thesh is not None:
            results = processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_sizes,
                threshold=conf_thesh,
            )
        else:
            results = processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_sizes,
            )
        # print(results)
        # <---- Perform results transformation (from swinb style modek outputs to coco style model outputs so the rest of the MAE calculation code can stay the same)
        for b in range(B):

            image_id = img_ids[b]

            # ---------------- GT merged mask ----------------
            gt_inst = inst_masks[b].to(device)
            if gt_inst.dim() == 4:
                gt_inst = gt_inst.squeeze(1)
            elif gt_inst.dim() != 3:
                raise ValueError(f"Unexpected GT shape: {gt_inst.shape}")

            if gt_inst.numel() == 0:
                continue

            gt_bin = (gt_inst > 0).float()     # (N_gt,H,W)
            merged_gt = gt_bin.max(dim=0)[0]      # (H,W)

            # ---------------- Pred masks ----------------
            segmentation_map = results[b]["segmentation"]
            segments_info = results[b]["segments_info"]
            seg = torch.as_tensor(segmentation_map, device=device)
            pred_masks = [(seg == seg_info["id"]).float()
                        for seg_info in segments_info]
            # print(pred_masks[0])


            if len(pred_masks) == 0:
                # print(f"[SKIPPED] No predictions for image {image_id}")
                continue

            pred_stack = torch.stack(pred_masks, dim=0)  # (P,H,W)

            # ---------------- IoU filter ----------------
            # IoU(P,H,W) × GT(N,H,W) → (P,N)
            ious = batch_iou_masks(pred_stack, gt_bin)
            # print(ious)
            best_iou_per_pred, _ = ious.max(dim=1)

            # Keep predictions that overlap with ANY GT
            keep_mask = best_iou_per_pred > 0.5
            if not keep_mask.any():
                # print(f"[SKIPPED MAE] No pred mask matched GT (IoU > 0.5) → {image_id}")
                continue

            valid_pred_masks = pred_stack[keep_mask]

            # ---------------- Merge kept predictions ----------------
            merged_pred = valid_pred_masks.max(dim=0)[0]  # (H,W)
            
            merged_pred = (merged_pred > 0).float()



            # ---------------- Per-image MAE ----------------
            mae_img = torch.abs(merged_pred - merged_gt).mean()

            total_val_mae += mae_img
            num_val_mae_samples += 1
            # print(f"{total_val_mae/num_val_mae_samples}")



            # ---------------- Match predicted masks to precomputed GT object masks ----------------
            if save_pred_masks: 
                if image_id in precomputed_gt_masks:
                    gt_obj_masks = precomputed_gt_masks[image_id]  # (N_obj,H,W) on GPU

                    if gt_obj_masks.numel() > 0 and len(pred_masks) > 0:

                        # IoU between all predicted masks and all GT object masks (GPU)
                        ious_po = batch_iou_masks(pred_stack, gt_obj_masks)  # (P,N_obj)

                        # For each GT object, find best matching predicted mask
                        best_iou_per_gt, best_pred_idx = ious_po.max(dim=0)  # (N_obj,)

                        obj_list = gt_data[image_id]["object_data"]

                        for obj_idx, obj in enumerate(obj_list):

                            total_objs += 1

                            if obj_idx >= best_iou_per_gt.numel():
                                break

                            iou_val = float(best_iou_per_gt[obj_idx].item())
                            pred_idx = int(best_pred_idx[obj_idx].item())

                            # Always store IoU value
                            obj["best_pred_iou"] = iou_val

                            # # If IoU <= 0.5 → no valid match
                            # if iou_val <= 0.5:
                            #     obj["pred_segmentation"] = None
                            #     print(f"No predicted mask matched GT object {obj_idx} in image {image_id}")
                            #     no_seg_mask_obj += 1
                            #     continue

                            # Convert matched predicted mask to COCO polygons
                            best_pred_mask = pred_stack[pred_idx]  # GPU mask (H,W)
                            best_pred_mask_cpu = (best_pred_mask > 0.8).float().cpu()

                            pred_polygons = mask_to_coco_polygon(best_pred_mask_cpu)

                            obj["pred_segmentation"] = pred_polygons



    avg_val_mae = total_val_mae / max(1, num_val_mae_samples)
    print(f"Backbone: {backbone} confidence theshold: {conf_thesh}")
    print(f"VAL MAE: {avg_val_mae:.4f}")
    # print(f"Predicted masks not found for {no_seg_mask_obj}/{total_objs}")

if save_pred_masks:
    # Save updated gt_data (with best_pred_iou/best_pred_index)
    with open(saved_data_path, "w") as f:
        json.dump(gt_data, f, indent=2)
