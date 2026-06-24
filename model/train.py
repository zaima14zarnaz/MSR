import sys
import os
import math
import shutil
import tempfile
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

# Component analysis: BSD variant -> precomputed features -> msr_model.msr MSRModel
from msr_model.salient_region_extraction_network import SalientRegionExtractionNetwork
from msr_model.msr import MSRModel
from losses import compute_losses

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pre_process.dataloader import SaliencyDataset
from metrics import filter_rois
from metrics import calculate_metrics
from pre_process.collate import variable_collate_fn

FEATURE_DIM = 256

device = "cpu"  # torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def _align_per_image_list(items, batch_size, empty_tensor):
    """Pad/truncate per-image lists so len == batch_size (one entry per image)."""
    aligned = []
    for b in range(batch_size):
        if b < len(items):
            aligned.append(items[b])
        else:
            aligned.append(empty_tensor)
    return aligned


def run_component_analysis_batch(bsd_model, rank_model, imgs, rois, phrases=None):
    """BSD -> ROI embeddings; MSR rank model (GAT + rank head) on precomputed features."""
    batch_size = imgs.shape[0]
    roi_embed, pred_masks, pred_class, phrase_saliency_scores = bsd_model(
        imgs, rois, phrases=phrases
    )
    rank_out = rank_model(roi_embed, rois=rois, phrases=None)

    empty_rank = roi_embed.new_zeros(0) if roi_embed.numel() > 0 else imgs.new_zeros(0)
    pred_ranks_per_image = _align_per_image_list(
        rank_out["rank_score"], batch_size, empty_rank
    )

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
        "rank_score": pred_ranks_per_image,
        "mask": pred_masks,
        "class_logits": pred_class_per_image,
        "phrase_saliency_score": phrase_saliency_scores,
    }


def component_analysis_checkpoint(bsd_model, rank_model):
    return {
        "bsd_model": bsd_model.state_dict(),
        "rank_model": rank_model.state_dict(),
    }
splits = {"train": "train", "val": "val", "test": "test"}
dataset_dir = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR"
# dataset_dir = "/data/research/zaima/dataset/Dataset/SIFR/SIFR_dataset"
# dataset_dir = "/data/research/zaima/dataset/Dataset/ASSR"
images_store = "images"
ranks_order_store = "rank_order"


def freeze_all_except_mask_head(model):
    for name, param in model.named_parameters():
        if "mask_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


# === Load both original train and val datasets ===
train_dataset = SaliencyDataset(
    image_dir=os.path.join(dataset_dir, images_store, splits["train"]),
    rank_dir=os.path.join(dataset_dir, ranks_order_store, splits["train"]),
    obj_seg_json=os.path.join(dataset_dir, f"obj_seg_data_{splits['train']}.json"),
    img_size=(512, 512),
    descriptions_csv=os.path.join(dataset_dir, "train_gpt4v.csv"),
)

val_dataset = SaliencyDataset(
    image_dir=os.path.join(dataset_dir, images_store, splits["val"]),
    rank_dir=os.path.join(dataset_dir, ranks_order_store, splits["val"]),
    obj_seg_json=os.path.join(dataset_dir, f"obj_seg_data_{splits['val']}.json"),
    img_size=(512, 512),
    descriptions_csv=os.path.join(dataset_dir, "train_gpt4v.csv"),
)

# === Combine / select samples from train (+ optionally val) ===
full_dataset = ConcatDataset([train_dataset, val_dataset])
# Use only the first 500 samples from the train split for this run
# full_dataset = Subset(train_dataset, list(range(500)))
# full_dataset = train_dataset

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = "/data/research/zaima/dataset/Dataset/msr-checkpoints"
model_save_path = os.path.join(model_save_path, timestamp)
os.makedirs(model_save_path, exist_ok=True)


def safe_torch_save(obj, path: str) -> None:
    """
    Write ``torch.save`` to a temp file in the same directory, then ``os.replace``
    to the final path. Avoids leaving a torn/corrupt ``.pth`` if the volume fills
    or the write is interrupted mid-stream.
    """
    path = os.path.normpath(os.path.abspath(path))
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".pth.tmp", dir=directory, text=False)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        try:
            du = shutil.disk_usage(directory)
            free_gib = du.free / (1024.0 ** 3)
        except OSError:
            free_gib = float("nan")
        raise RuntimeError(
            f"Checkpoint save failed ({path}): {e}. "
            f"Check disk space / quota; free space on save volume ≈ {free_gib:.2f} GiB."
        ) from e


def train(model_save_path, run_no, full_dataset, seed=42):
    # === Resplit 90% train / 10% val ===
    total_size = len(full_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    new_train_dataset, new_val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),  # ensures reproducibility
    )

    # === Create DataLoaders ===
    train_loader = DataLoader(
        new_train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=variable_collate_fn,
    )
    val_loader = DataLoader(
        new_val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=variable_collate_fn,
    )

    bsd_model = SalientRegionExtractionNetwork(
        backbone_pretrained=True, mod_injection=False, dropout_p=0.2
    ).to(device)
    rank_model = MSRModel(feature_dim=FEATURE_DIM, dropout_p=0.2).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": bsd_model.parameters(), "lr": 1e-5},
            # {"params": rank_model.gat.parameters(), "lr": 1e-5},
            {"params": rank_model.rank_head.parameters(), "lr": 1e-5},
        ]
    )

    num_epochs = 7
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )
    criterion_cls = torch.nn.CrossEntropyLoss()
    overlap_threshold = 0.5
    confidence_threshold = 0.5

    best_loss = float("inf")
    best_sor = 0
    mask_quality_debug = True
    for epoch in range(num_epochs):
        if epoch == 2:
            break
        bsd_model.train()
        rank_model.train()
        (
            train_loss_sum,
            train_rho_sum,
            n_rho,
            valid_train_rhos,
            train_sasor_sum,
            n_sasor,
            train_mae_total,
            n_mae,
        ) = (0.0, 0.0, 0, 0, 0.0, 0, 0.0, 0)
        mask_loss_train = 0.0
        phrase_overlay_train = 0.0
        phrase_corr_train_sum = 0.0
        n_phrase_corr_train = 0

        pbar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stdout.isatty(),
        )
        mask_saved = False

        for image_ids, imgs, gts, gt_ranks, gt_obj_class, img_mask, masks, rois, phrases in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            img_mask = img_mask.to(device)

            obj_masks = [m.to(device) for m in masks]
            valid_rois = [r for r in rois if r.numel() > 0]
            rois = torch.cat(rois, dim=0).to(device)

            gt_obj_class = torch.cat(gt_obj_class, dim=0).to(device)

            phrase_list = list(phrases)

            optimizer.zero_grad()
            out = run_component_analysis_batch(
                bsd_model, rank_model, imgs, rois, phrases=phrase_list
            )
            pred_ranks = out["rank_score"]
            pred_masks = out["mask"]
            pred_class = out["class_logits"]
            phrase_saliency_scores = out["phrase_saliency_score"]

            if mask_quality_debug:
                if not mask_saved:
                    mask_saved = True
                    mask = pred_masks[0]
                    if mask.dim() == 3 and mask.size(0) == 1:
                        mask = mask.squeeze(0)
                    mask_img = (mask.detach().cpu().clamp(0, 1) * 255).byte()
                    mask_pil = TF.to_pil_image(mask_img)
                    os.makedirs("outputs", exist_ok=True)
                    mask_pil.save("outputs/pred_mask.png")

                    gt = img_mask[0].detach().cpu()
                    gt = gt.squeeze()
                    assert gt.dim() == 2, f"GT mask has wrong shape: {gt.shape}"
                    gt_vis = (gt >= 0.001).float() * 255
                    gt_vis = gt_vis.byte()
                    TF.to_pil_image(gt_vis).save("outputs/gt_mask.png")

            filtered_pred_ranks, filtered_pred_class, filtered_pred_masks, filtered_indices, _ = filter_rois(
                img_mask,
                pred_masks,
                pred_ranks,
                pred_class,
                device=device,
                overlap_threshold=overlap_threshold,
                confidence_threshold=confidence_threshold,
            )

            with torch.no_grad():
                metrics = calculate_metrics(
                    gt_ranks,
                    pred_ranks,
                    filtered_pred_ranks,
                    obj_masks,
                    filtered_indices=filtered_indices,
                    pred_masks=pred_masks,
                    seg_pred_masks=None,
                    gt_masks=gts,
                )
                train_rho_sum += metrics["sor_sum"]
                n_rho += metrics["valid_sor_count"]
                valid_train_rhos += metrics["non_issue_rhos"]
                train_sasor_sum += metrics["sa_sor_sum"]
                n_sasor += metrics["valid_sa_sor_count"]
                train_mae_total += metrics["mae_sum"]
                n_mae += metrics["valid_mae_count"]

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
                mask_loss_train += mask_loss.detach().item()
            else:
                mask_loss_train += float(mask_loss)
            if isinstance(phrase_overlay_loss, torch.Tensor):
                phrase_overlay_train += phrase_overlay_loss.detach().item()
            else:
                phrase_overlay_train += float(phrase_overlay_loss)
            batch_corr = losses_dict.get("phrase_rank_correlation", float("nan"))
            if isinstance(batch_corr, torch.Tensor):
                batch_corr = batch_corr.detach().item()
            if math.isfinite(batch_corr):
                phrase_corr_train_sum += float(batch_corr)
                n_phrase_corr_train += 1

            if rank_loss == 0:
                continue

            total_loss.backward()
            optimizer.step()

            train_loss_sum += total_loss.item()

        scheduler.step()

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_mask_loss = mask_loss_train / len(train_loader)
        avg_train_phrase_overlay = phrase_overlay_train / len(train_loader)
        avg_train_phrase_corr = (
            phrase_corr_train_sum / n_phrase_corr_train
            if n_phrase_corr_train > 0
            else float("nan")
        )
        avg_train_rho = (1 + (train_rho_sum / n_rho if n_rho > 0 else 0)) / 2
        avg_train_sasor = train_sasor_sum / n_sasor if n_sasor > 0 else 0
        avg_train_mae = train_mae_total / n_mae if n_mae > 0 else 0

        # ---------------- VALIDATION ----------------
        bsd_model.eval()
        rank_model.eval()
        (
            val_loss_sum,
            val_rho_sum,
            n_rho,
            valid_val_rhos,
            val_sasor_sum,
            n_sasor,
            val_mae_total,
            n_mae,
        ) = (0.0, 0.0, 0, 0, 0.0, 0, 0.0, 0)
        mask_loss_val = 0.0
        phrase_overlay_val = 0.0
        phrase_corr_val_sum = 0.0
        n_phrase_corr_val = 0

        pbar = tqdm(
            val_loader,
            desc=f"Validating Epoch {epoch+1}/{num_epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stdout.isatty(),
        )

        with torch.no_grad():
            for image_ids, imgs, gts, gt_ranks, gt_obj_class, img_mask, masks, rois, phrases in pbar:
                imgs, gts = imgs.to(device), gts.to(device)
                img_mask = img_mask.to(device)

                obj_masks = [m.to(device) for m in masks]
                valid_rois = [r for r in rois if r.numel() > 0]
                rois = torch.cat(rois, dim=0).to(device)

                gt_obj_class = torch.cat(gt_obj_class, dim=0).to(device)
                phrase_list = list(phrases)

                out = run_component_analysis_batch(
                bsd_model, rank_model, imgs, rois, phrases=phrase_list
            )
                pred_ranks = out["rank_score"]
                pred_masks = out["mask"]
                pred_class = out["class_logits"]
                phrase_saliency_scores = out["phrase_saliency_score"]

                filtered_pred_ranks, filtered_pred_class, filtered_pred_masks, filtered_indices, _ = filter_rois(
                    img_mask,
                    pred_masks,
                    pred_ranks,
                    pred_class,
                    device=device,
                    overlap_threshold=overlap_threshold,
                    confidence_threshold=confidence_threshold,
                )

                metrics = calculate_metrics(
                    gt_ranks,
                    pred_ranks,
                    filtered_pred_ranks,
                    masks,
                    filtered_indices=filtered_indices,
                    pred_masks=pred_masks,
                    gt_masks=gts,
                )
                val_rho_sum += metrics["sor_sum"]
                n_rho += metrics["valid_sor_count"]
                valid_val_rhos += metrics["non_issue_rhos"]
                val_sasor_sum += metrics["sa_sor_sum"]
                n_sasor += metrics["valid_sa_sor_count"]
                val_mae_total += metrics["mae_sum"]
                n_mae += metrics["valid_mae_count"]

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
                if isinstance(mask_loss, torch.Tensor):
                    mask_loss_val += mask_loss.detach().item()
                else:
                    mask_loss_val += float(mask_loss)
                if isinstance(phrase_overlay_loss, torch.Tensor):
                    phrase_overlay_val += phrase_overlay_loss.detach().item()
                else:
                    phrase_overlay_val += float(phrase_overlay_loss)
                batch_corr = losses_dict.get("phrase_rank_correlation", float("nan"))
                if isinstance(batch_corr, torch.Tensor):
                    batch_corr = batch_corr.detach().item()
                if math.isfinite(batch_corr):
                    phrase_corr_val_sum += float(batch_corr)
                    n_phrase_corr_val += 1
                if float(rank_loss) == 0.0:
                    continue
                total_loss = losses_dict["total_loss"]
                val_loss_sum += total_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_mask_loss = mask_loss_val / len(val_loader)
        avg_val_phrase_overlay = phrase_overlay_val / len(val_loader)
        avg_val_phrase_corr = (
            phrase_corr_val_sum / n_phrase_corr_val
            if n_phrase_corr_val > 0
            else float("nan")
        )
        avg_val_sor = (1 + (val_rho_sum / n_rho if n_rho > 0 else 0)) / 2
        avg_val_sasor = val_sasor_sum / n_sasor if n_sasor > 0 else 0
        avg_val_mae = val_mae_total / n_mae if n_mae > 0 else 0

        print(
            f"Epoch [{epoch+1}/{num_epochs}] (Seed: {seed}) \n"
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | \n"
            f"Train Phrase-overlay loss: {avg_train_phrase_overlay:.4f} | Val Phrase-overlay loss: {avg_val_phrase_overlay:.4f} | \n"
            f"Train Phrase–rank Spearman ρ: {avg_train_phrase_corr:.4f} | Val Phrase–rank Spearman ρ: {avg_val_phrase_corr:.4f} | \n"
            f"Train Mask Loss: {avg_train_mask_loss:.4f} | Val Mask Loss: {avg_val_mask_loss:.4f} | \n"
            f"Train SOR (ρ): {avg_train_rho:.4f} ({valid_train_rhos}/{len(train_loader) * 4})| Val SOR (ρ): {avg_val_sor:.4f} ({valid_val_rhos}/{n_rho}) | \n"
            f"Train SA-SOR : {avg_train_sasor:.4f} | Val SA-SOR : {avg_val_sasor:.4f} | \n"
            f"Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_name = f"best_loss_run_{run_no}.pth"
            safe_torch_save(
                component_analysis_checkpoint(bsd_model, rank_model),
                os.path.join(model_save_path, best_model_name),
            )
            print(f"✅ Saved new best model (val loss: {best_loss:.4f})")
        if avg_val_sor > best_sor:
            best_sor = avg_val_sor
            best_model_name = f"best_sor_run_{run_no}.pth"
            safe_torch_save(
                component_analysis_checkpoint(bsd_model, rank_model),
                os.path.join(model_save_path, best_model_name),
            )
            print(f"✅ Saved new best model (val sor: {best_sor:.4f})")
        model_name = f"epoch_{epoch+1}_{run_no}_{str(avg_val_sor)[:6].replace('.', '-')}.pth"
        safe_torch_save(
            component_analysis_checkpoint(bsd_model, rank_model),
            os.path.join(model_save_path, model_name),
        )

    return best_sor


runs = 1
sors = []
for i in range(0, runs):
    best_sor = train(
        model_save_path=model_save_path, run_no=i, full_dataset=full_dataset, seed=42
    )
    sors.append(best_sor)
    print()

print(f"Best SOR score across {runs} runs: {sum(sors)/len(sors):.4f}")
