import sys
import os
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision
from torchvision.utils import draw_bounding_boxes
import numpy as np
from torch.utils.data import random_split, ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
from scipy.stats import spearmanr
from datetime import datetime
import time


# from bsd.backbone_variants.variant_2A_res_50 import SalientRegionExtractionNetwork
# from tspnet_model.tspnet import TSPNet
# from bsd.visual_prompt_experiment.backbones.res101 import SalientRegionExtractionNetwork
# from tspnet_model.tspnet import TSPNet
from tspnet_model.salient_region_extraction_network import SalientRegionExtractionNetwork
from tspnet_model.tspnet import TSPNet
from losses import compute_losses

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pre_process.dataloader import SaliencyDataset
from metrics import filter_rois
from metrics import calculate_metrics
from pre_process.collate import variable_collate_fn

FEATURE_DIM = 256


def run_component_analysis_batch(bsd_model, rank_model, imgs, rois):
    batch_size = imgs.shape[0]
    roi_embed, pred_masks, pred_class, _ = bsd_model(imgs, rois, phrases=None)
    rank_out = rank_model(roi_embed, rois=rois, phrases=None)

    if pred_class is not None and pred_class.numel() > 0 and rois.numel() > 0:
        batch_idx = rois[:, 0].long()
        pred_class_per_image = [
            pred_class[batch_idx == b] for b in range(batch_size)
        ]
    else:
        num_classes = pred_class.shape[-1] if pred_class is not None else 90
        pred_class_per_image = [
            imgs.new_zeros((0, num_classes)) for _ in range(batch_size)
        ]

    return {
        "rank_score": rank_out["rank_score"],
        "mask": pred_masks,
        "class_logits": pred_class_per_image,
        "phrase_saliency_score": rank_out["phrase_saliency_score"],
    }


def load_component_analysis_checkpoint(bsd_model, rank_model, state):
    if isinstance(state, dict) and "bsd_model" in state and "rank_model" in state:
        bsd_model.load_state_dict(state["bsd_model"])
        rank_model.load_state_dict(state["rank_model"])
        return
    raise RuntimeError(
        "Expected checkpoint with top-level keys 'bsd_model' and 'rank_model' "
        "(component-analysis format from train_contra_loss.py)."
    )


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
splits = {"train": "train", "val": "val", "test": "test"}
dataset_dir = "../Dataset/IRSR_ASSR"
# dataset_dir = "../Dataset/ASSR" 
# dataset_dir = "../Dataset/SIFR_dataset"
sal_map_dir = "../saliency_maps"
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
# test_dataset = Subset(test_dataset, list(range(50)))
test_loader = DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=variable_collate_fn
)

model_dir = "../tspnet-checkpoints/20260520_141257/epoch_2_0_0-8821.pth"
state = torch.load(model_dir, map_location="cpu")
bsd_model = SalientRegionExtractionNetwork(
    backbone_pretrained=True, mod_injection=False, dropout_p=0.2
).to(device)
rank_model = TSPNet(feature_dim=FEATURE_DIM, dropout_p=0.2).to(device)
load_component_analysis_checkpoint(bsd_model, rank_model, state)
bsd_model.eval()
rank_model.eval()


val_loss_sum, val_rho_sum, n_rho, valid_val_rhos, val_sasor_sum, n_sasor = 0.0, 0.0, 0, 0, 0.0, 0
mask_loss_val = 0.0
phrase_overlay_val_sum = 0.0
phrase_corr_val_sum = 0.0
n_phrase_corr_val = 0
n_total_loss_batches = 0
mae_total, n_mae = 0.0, 0
criterion_cls = torch.nn.CrossEntropyLoss()
overlap_threshold = 0.5
# confidence_threshold  = 0.5 # For IRSR and SIFR dataset
confidence_threshold = 0.4  # For ASSR dataset (vis only)

pbar = tqdm(
    test_loader,
            desc=f"Evaluating",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stdout.isatty()
    )
start_time = time.time()
with torch.no_grad():
    for img_ids, imgs, gts, gt_ranks, gt_obj_class, img_mask, inst_masks, rois, phrases in pbar:

        imgs = imgs.to(device)
        gts = gts.to(device)

        # Convert lists to tensors for segmentation loss
        img_mask = torch.stack([m.to(device) for m in img_mask], dim=0)   # (B,1,H,W)
        inst_masks = [m.to(device) for m in inst_masks]                   # list of (N_i,1,H,W)

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
        out = run_component_analysis_batch(bsd_model, rank_model, imgs, rois)
        pred_ranks = out["rank_score"]
        pred_masks = out["mask"]
        pred_class = out["class_logits"]
        phrase_saliency_scores = out["phrase_saliency_score"]

        # -------------------------------
        # Segmentation Model Forward
        # -------------------------------
       

        # -------------------------------
        # Rank Model Overlap Filter
        # -------------------------------
        filtered_pred_ranks, filtered_pred_class, filtered_pred_masks, filtered_indices, _ = filter_rois(
            img_mask, pred_masks, pred_ranks, pred_class, device=device,
            overlap_threshold=overlap_threshold,
            confidence_threshold=confidence_threshold
        )

        # -------------------------------
        # Metrics
        # -------------------------------
        metrics = calculate_metrics(gt_ranks, 
                                            pred_ranks, 
                                            filtered_pred_ranks, 
                                            inst_masks, 
                                            filtered_indices=filtered_indices, 
                                            pred_masks=pred_masks, 
                                            seg_pred_masks=None,
                                            gt_masks=gts)

        val_rho_sum        += metrics["sor_sum"]
        n_rho              += metrics["valid_sor_count"]
        valid_val_rhos     += metrics["non_issue_rhos"]
        val_sasor_sum      += metrics["sa_sor_sum"]
        n_sasor            += metrics["valid_sa_sor_count"]
        mae_total          += metrics["mae_sum"]
        n_mae              += metrics["valid_mae_count"]

        losses_dict = compute_losses(
            gt_ranks,
            img_mask,
            gt_obj_class,
            pred_ranks,
            pred_masks,
            pred_class,
            phrase_saliency_scores,
            valid_rois,
            filtered_pred_class,
            criterion_cls,
            device,
        )
        rank_loss = losses_dict["rank_loss"]
        mask_loss = losses_dict["mask_loss"]
        class_loss = losses_dict["class_loss"]
        phrase_overlay_loss = losses_dict["phrase_overlay_loss"]
        total_loss = losses_dict["total_loss"]
        if isinstance(mask_loss, torch.Tensor):
            mask_loss_val += mask_loss.detach().item()
        else:
            mask_loss_val += float(mask_loss)
        if isinstance(phrase_overlay_loss, torch.Tensor):
            phrase_overlay_val_sum += phrase_overlay_loss.detach().item()
        else:
            phrase_overlay_val_sum += float(phrase_overlay_loss)
        batch_corr = losses_dict.get("phrase_rank_correlation", float("nan"))
        if isinstance(batch_corr, torch.Tensor):
            batch_corr = batch_corr.detach().item()
        if math.isfinite(batch_corr):
            phrase_corr_val_sum += float(batch_corr)
            n_phrase_corr_val += 1
        if float(rank_loss) != 0.0:
            val_loss_sum += total_loss.detach().item()
            n_total_loss_batches += 1

        # # -------------------------------
        # # Saliency Maps per-image
        # # -------------------------------
        # for obj_idx, m in enumerate(per_obj_pred_masks[0]):
        #     save_sal_map_gray(m > 0.5, f"debug_predmask_img{img_ids[0]}_obj{obj_idx}.png")


        # sal_maps = create_sal_maps(gts, per_obj_pred_masks, pred_ranks)

        # for img_id, sal_map in zip(img_ids, sal_maps):
        #     fname = os.path.join(sal_map_dir, f"{img_id}.png")
        #     save_sal_map_gray(sal_map, fname)
        #     print(f"Saved saliency map for {img_id}.png")


avg_val_sor = (1 + (val_rho_sum / n_rho if n_rho > 0 else 0)) / 2
avg_val_sasor = val_sasor_sum / n_sasor if n_sasor > 0 else 0
avg_val_mae = mae_total / n_mae if n_mae > 0 else 0
avg_val_mask_loss = mask_loss_val / len(test_loader)
avg_val_phrase_overlay = phrase_overlay_val_sum / len(test_loader)
avg_val_phrase_corr = (
    phrase_corr_val_sum / n_phrase_corr_val if n_phrase_corr_val > 0 else float("nan")
)
avg_val_total_loss = val_loss_sum / n_total_loss_batches if n_total_loss_batches > 0 else 0.0

if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.time()
total_inference_time = end_time - start_time

print(f"Dataset Name: {dataset_name} | confidence threshold: {confidence_threshold}")
print(
    f"Test SOR (ρ): {avg_val_sor:.4f} ({valid_val_rhos}/{len(test_loader)*4})  |  "
    f"SA-SOR (ρ): {avg_val_sasor:.4f} ({n_sasor}/{len(test_loader)*4}) | MAE: {avg_val_mae:.4f} | "
    f"Phrase–rank Spearman ρ: {avg_val_phrase_corr:.4f} | "
    f"Time: {total_inference_time}s"
)
print(f"Weight path: {model_dir}")
