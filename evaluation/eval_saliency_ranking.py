import sys
import os
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
import json

from seg_utility import mask_to_coco_polygon
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.bsd.backbone_variants.variant_1A_res_101 import SalientRegionExtractionNetwork
from model.variants.variant_J import LGSRModel
from model.losses import compute_losses
from metrics import filter_rois
from metrics import calculate_metrics
from pre_process.dataloader import SaliencyDataset
from pre_process.collate import variable_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits = {"train": "train", "val": "val", "test": "test"}
dataset_dir = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR"
# dataset_dir = "/data/research/zaima/dataset/Dataset/ASSR" 
# dataset_dir = "/data/research/zaima/dataset/Dataset/SIFR_dataset"
sal_map_dir = "/home/zaimaz/Desktop/research1/QAGNet/sr-model1/saliency_maps"
rank_data_json = "sr_ranks.json"
os.makedirs(sal_map_dir, exist_ok=True)
dataset_name = os.path.basename(dataset_dir)
images_store = "images"
ranks_order_store = "rank_order"
sal_map_store = "gt"

test_dataset = SaliencyDataset(
    image_dir=os.path.join(dataset_dir, images_store, splits["test"]),
    rank_dir=os.path.join(dataset_dir, ranks_order_store, splits["test"]),
    obj_seg_json=os.path.join(dataset_dir, f"obj_seg_data_{splits['test']}.json"),
    img_size=(512, 512),
    descriptions_csv = os.path.join(dataset_dir, "test.csv")
)

test_loader = DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=variable_collate_fn
)

model_dir = "/home/zaimaz/Desktop/research1/QAGNet/sr-model1/weights/res101-vis-text-pos-4A-J-irsr.pth"
state = torch.load(model_dir, map_location="cpu")
bsd_model = SalientRegionExtractionNetwork(backbone_pretrained=True)
bsd_model = bsd_model.to(device)  
model = LGSRModel(salient_region_extractor=bsd_model).to(device)
model.load_state_dict(state)
model.eval()


val_loss_sum, val_rho_sum, n_rho, valid_val_rhos, val_sasor_sum, n_sasor = 0.0, 0.0, 0, 0, 0.0, 0
mask_loss_val = 0.0
mae_total, n_mae = 0.0, 0
overlap_threshold = 0.5
confidence_threshold = 0.5
height, width = 512, 512
gt_data = {}

pbar = tqdm(
    test_loader,
    desc=f"Evaluating",
    unit="batch",
    dynamic_ncols=True,
    leave=False,
    disable=not sys.stdout.isatty()
)

with torch.no_grad():
    for img_ids, imgs, gts, gt_ranks, gt_obj_class, img_mask, inst_masks, rois, phrases in pbar:

        imgs = imgs.to(device)
        gts = gts.to(device)

        # Convert lists to tensors for segmentation loss
        img_mask = torch.stack([m.to(device) for m in img_mask], dim=0)   # (B,1,H,W)
        inst_masks = [m.to(device) for m in inst_masks]                   # list of (N_i,1,H,W) or (N_i,H,W)

        # -------------------------------
        # Rank Model Forward
        # -------------------------------
        valid_rois = [r for r in rois if r.numel() > 0]
        valid_classes = [c for c in gt_obj_class if c.numel() > 0]

        if any(r.numel() for r in valid_rois):
            rois = torch.cat(valid_rois, dim=0).to(device)
            gt_obj_class = torch.cat(valid_classes, dim=0).to(device)
        else:
            rois = torch.zeros((0, 5), dtype=torch.float32, device=device)
            gt_obj_class = torch.zeros((0,), dtype=torch.long, device=device)

        phrase_list = list(phrases)

        out = model(imgs, rois=rois, phrases=phrases)
        pred_ranks = out["rank_score"]   # should be iterable per image (as expected by calculate_sor)
        pred_masks = out["mask"]
        pred_class = out["class_logits"]

        # -------------------------------
        # Rank Model Overlap Filter
        # -------------------------------
        filtered_pred_ranks, filtered_pred_class, filtered_pred_masks, filtered_indices, _ = filter_rois(
            img_mask, pred_masks, pred_ranks, pred_class, device=device,
            overlap_threshold=overlap_threshold,
            confidence_threshold=confidence_threshold
        )

        # Make sure filtered_indices is a plain Python list for membership tests
        filtered_indices_list = [int(i) for i in filtered_indices]

        # -------------------------------
        # Metrics
        # -------------------------------
        metrics = calculate_metrics(
            gt_ranks,
            pred_ranks,
            filtered_pred_ranks,
            inst_masks,
            filtered_indices=filtered_indices_list,
            pred_masks=pred_masks,
            seg_pred_masks=None,
            gt_masks=gts
        )

        val_rho_sum        += metrics["sor_sum"]
        n_rho              += metrics["valid_sor_count"]
        valid_val_rhos     += metrics["non_issue_rhos"]
        val_sasor_sum      += metrics["sa_sor_sum"]
        n_sasor            += metrics["valid_sa_sor_count"]
        mae_total          += metrics["mae_sum"]
        n_mae              += metrics["valid_mae_count"]

        # -----------------------------------------------------------
        # Build gt_data_list with SAME LOGIC as calculate_sor
        # -----------------------------------------------------------
        # gt_ranks: iterable of GT rank tensors (per image)
        # pred_ranks: iterable of predicted score tensors (per image)
        # inst_masks: list of per-image instance masks (N_i,1,H,W or N_i,H,W)

        for image_id, pred, gt, img_inst_masks in zip(img_ids, pred_ranks, gt_ranks, inst_masks):

            # 1) Convert to numpy the same way as in calculate_sor
            p = pred.detach().cpu().numpy().flatten()
            gt_idx = gt.cpu().numpy().astype(int)

            # 2) Only include GT indices that exist in the prediction vector
            valid_idx = gt_idx[gt_idx < len(p)]

            if len(valid_idx) == 0:
                # No valid GT objects; still append empty entry for this image
                gt_data.append({
                    "image_id": image_id,
                    "object_data": []
                })
                continue

            # 3) Extract predicted scores at those GT positions
            pred_for_gt = p[valid_idx]

            # 4) Convert predicted scores to ranking order (same as calculate_sor)
            rank_pred = np.argsort(np.argsort(pred_for_gt))
            rank_gt   = np.arange(len(valid_idx))

            # 5) KEEP ONLY GT POSITIONS IN filtered_indices (exactly as calculate_sor)
            # keep_mask = np.array(
            #     [idx in filtered_indices_list for idx in valid_idx],
            #     dtype=bool
            # )
            # 5) Turn OFF the filtered_indices filtering
            keep_mask = np.ones_like(valid_idx, dtype=bool)

            rank_pred_kept = rank_pred[keep_mask]
            rank_gt_kept   = rank_gt[keep_mask]
            kept_idx       = valid_idx[keep_mask]

            object_data = []

            # For each kept GT object, attach segmentation + ranks
            for inst_id, gt_rank_val, pred_rank_val in zip(kept_idx, rank_gt_kept, rank_pred_kept):

                # img_inst_masks shape can be (N_i,1,H,W) or (N_i,H,W)
                seg = img_inst_masks[inst_id]
                if seg.dim() == 3:      # (1,H,W) → (H,W)
                    seg_mask = seg.squeeze(0)
                elif seg.dim() == 2:    # (H,W) already
                    seg_mask = seg
                else:
                    # Fallback: best-effort squeeze to 2D
                    seg_mask = seg.squeeze()

                object_data.append({
                    "segmentation": mask_to_coco_polygon(seg_mask),        # tensor (H,W)
                    "gt_rank": int(gt_rank_val),          # integer rank used in Spearman
                    "pred_rank": int(pred_rank_val)       # integer rank used in Spearman
                })

            gt_data[image_id] = {
                "image_id": image_id,
                "object_data": object_data
            }
            # print(gt_data_list[0])



                        

avg_val_sor = (1 + (val_rho_sum / n_rho if n_rho > 0 else 0)) / 2
avg_val_sasor = val_sasor_sum / n_sasor if n_sasor > 0 else 0
avg_val_mae = mae_total / n_mae if n_mae > 0 else 0

print(f"Dataset Name: {dataset_name}")
print(f"Test SOR (ρ): {avg_val_sor:.4f} ({valid_val_rhos}/{len(test_loader)*4})  |  SA-SOR (ρ): {avg_val_sasor:.4f} ({n_sasor}/{len(test_loader)*4}) | MAE: {avg_val_mae:.4f}")
print(f"Weight path: {model_dir}")



with open(rank_data_json, "w") as f:
    json.dump(gt_data, f, indent=2)