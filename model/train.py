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

# from bsd.bsd import MultiScaleSaliency
# from bsd.roi_feats_fusion_adaptive import MultiScaleSaliencyMAFormer
# from bsd.backbone_variants.variant_4A_swinL import SalientRegionExtractionNetwork
from bsd.backbone_variants.variant_4A_swinL import SalientRegionExtractionNetwork
from variants.variant_I import LGSRModel
from losses import compute_losses
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_process.dataloader import SaliencyDataset
from metrics import filter_rois
from metrics import calculate_metrics
from pre_process.collate import variable_collate_fn


device = "cuda:1" # torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    # descriptions_csv=os.path.join(dataset_dir,"train.csv")
)

val_dataset = SaliencyDataset(
    image_dir=os.path.join(dataset_dir, images_store, splits["val"]),
    rank_dir=os.path.join(dataset_dir, ranks_order_store, splits["val"]),
    obj_seg_json=os.path.join(dataset_dir, f"obj_seg_data_{splits['val']}.json"),
    img_size=(512, 512),
    # descriptions_csv = os.path.join(dataset_dir, "train.csv")
)

# === Combine all samples from train + val ===
full_dataset = ConcatDataset([train_dataset, val_dataset])
# full_dataset = train_dataset

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = "/home/zaimaz/Desktop/research1/QAGNet/sr-model1/checkpoints"
model_save_path = os.path.join(model_save_path, timestamp)
os.makedirs(model_save_path, exist_ok=True)

def train(model_save_path, run_no, full_dataset, seed=42):
    # === Resplit 90% train / 10% val ===
    total_size = len(full_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    new_train_dataset, new_val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)  # ensures reproducibility
    )

    # === Create DataLoaders ===
    train_loader = DataLoader(
        new_train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=variable_collate_fn
    )
    val_loader = DataLoader(
        new_val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=variable_collate_fn
    )


    
    bsd_model = SalientRegionExtractionNetwork(backbone_pretrained=True, film_injection=False, dropout_p=0.2)
    # bsd_model.load_state_dict(torch.load(bsd_weight_path, map_location=device))
    bsd_model = bsd_model.to(device)  
    # bsd_model.eval()                 
  

    model = LGSRModel(salient_region_extractor=bsd_model).to(device)
    # model.freeze_saliency()
    # model = SimplifiedSaliencyRankNet().to(device)
    optimizer = torch.optim.Adam([
        {'params': model.rank_head.parameters(), 'lr': 1e-5},
        # {'params': model.aggregator.parameters(), 'lr': 1e-5},
        {'params': model.salient_region_extractor.parameters(), 'lr': 1e-5}
    ])

    num_epochs = 7
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    criterion_cls = torch.nn.CrossEntropyLoss()
    overlap_threshold = 0.5
    confidence_threshold = 0.5

    best_loss = float("inf")  # smaller loss = better
    best_sor = 0
    mask_quality_debug = True
    for epoch in range(num_epochs):
        if epoch == 2:
            break
        # ---------------- TRAIN ----------------
        # # ----- Freeze backbone + all heads except mask head -----
        # if epoch == 2:      # Freeze starting from epoch 3
        #     freeze_all_except_mask_head(model)
        #     optimizer = torch.optim.Adam(
        #         filter(lambda p: p.requires_grad, model.parameters()),
        #         lr=1e-4   # or your previous LR
        #     )
        #     print(">>> All parameters frozen except mask_head")
        model.train()
        train_loss_sum, train_rho_sum, n_rho, valid_train_rhos, train_sasor_sum, n_sasor, train_mae_total, n_mae = 0.0, 0.0, 0, 0, 0.0, 0, 0.0, 0
        mask_loss_train = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stdout.isatty()
        )
        mask_saved = False

        # todo: Grab phrase tensor in the train loop and set their device to gpu
        for image_ids, imgs, gts, gt_ranks, gt_obj_class, img_mask, masks, rois, phrases in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            img_mask = img_mask.to(device)

            obj_masks = [m.to(device) for m in masks]
            valid_rois = [r for r in rois if r.numel() > 0]
            rois = torch.cat(rois, dim=0).to(device)

            gt_obj_class = torch.cat(gt_obj_class, dim=0).to(device)

            # ---------------------------------------------------------
            # 🔥 TEXT IS STILL PYTHON STRINGS → DO NOT `.to(device)`
            # ---------------------------------------------------------
            # Model will tokenize and move to device internally.
            phrase_list = list(phrases)
            # Example: [["A boy running", "Grass in background"], [...], ...]

            optimizer.zero_grad()

            out = model(imgs, rois=rois, phrases=phrase_list)
            pred_ranks = out["rank_score"]
            pred_masks = out["mask"]
            pred_class = out["class_logits"]



            # --- Save one predicted mask for visualization ---
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



            # --- Filter out low-overlap ROIs ---
            filtered_pred_ranks, filtered_pred_class, filtered_pred_masks, filtered_indices, _ = filter_rois(img_mask, 
                                                                                        pred_masks,
                                                                                        pred_ranks,
                                                                                        pred_class,
                                                                                        device=device,
                                                                                        overlap_threshold=overlap_threshold,
                                                                                        confidence_threshold=confidence_threshold)


            # -------- Metrics (Spearman) --------
            with torch.no_grad():               
                metrics = calculate_metrics(gt_ranks, 
                                            pred_ranks, 
                                            filtered_pred_ranks, 
                                            obj_masks, 
                                            filtered_indices=filtered_indices, 
                                            pred_masks=pred_masks, 
                                            seg_pred_masks=None,
                                            gt_masks=gts)
                train_rho_sum += metrics["sor_sum"]
                n_rho += metrics["valid_sor_count"]
                valid_train_rhos += metrics["non_issue_rhos"]
                train_sasor_sum += metrics["sa_sor_sum"]
                n_sasor += metrics["valid_sa_sor_count"]
                train_mae_total += metrics["mae_sum"]
                n_mae += metrics["valid_mae_count"]

            # -------- Loss Computation --------
            losses_dict = compute_losses(gt_ranks, 
                   img_mask, 
                   gt_obj_class, 
                   pred_ranks,
                   pred_masks, 
                   pred_class,
                   valid_rois, 
                   filtered_pred_class, 
                   criterion_cls, 
                   device)
            rank_loss = losses_dict["rank_loss"]
            mask_loss = losses_dict["mask_loss"]
            class_loss = losses_dict["class_loss"]
            if isinstance(mask_loss, torch.Tensor):
                mask_loss_train += mask_loss.detach().item()
            else:
                mask_loss_train += float(mask_loss)

            if rank_loss == 0:
                continue
            total_loss = rank_loss + mask_loss + class_loss

            # -------- Backprop --------
            total_loss.backward()
            optimizer.step()

            # -------- Tracking --------
            train_loss_sum += total_loss.item()


        scheduler.step()

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_mask_loss = mask_loss_train / len(train_loader)
        avg_train_rho = (1 + (train_rho_sum / n_rho if n_rho > 0 else 0)) / 2
        avg_train_sasor = train_sasor_sum / n_sasor if n_sasor > 0 else 0
        avg_train_mae = train_mae_total / n_mae if n_mae > 0 else 0
        


        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss_sum, val_rho_sum, n_rho, valid_val_rhos, val_sasor_sum, n_sasor, val_mae_total, n_mae = 0.0, 0.0, 0, 0, 0.0, 0, 0.0, 0
        mask_loss_val = 0.0


        pbar = tqdm(
            val_loader,
            desc=f"Validating Epoch {epoch+1}/{num_epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stdout.isatty()
        )

        with torch.no_grad():
            for image_ids, imgs, gts, gt_ranks, gt_obj_class, img_mask, masks, rois, phrases in pbar:
                imgs, gts = imgs.to(device), gts.to(device)
                img_mask = img_mask.to(device)

                # obj_masks vs masks comes from collate_fn
                obj_masks = [m.to(device) for m in masks]           # <-- corrected
                valid_rois = [r for r in rois if r.numel() > 0]
                rois = torch.cat(rois, dim=0).to(device)            # <-- correct ROI boxes

                gt_obj_class = torch.cat(gt_obj_class, dim=0).to(device)
                phrase_list = list(phrases)

                optimizer.zero_grad()
                

                # -------- Forward --------
                # todo: Send phrase set to model out = model(imgs, rois=rois, phrases=phrases)
                out = model(imgs, rois=rois, phrases=phrase_list)
                pred_ranks = out["rank_score"]
                pred_masks = out["mask"]
                pred_class = out["class_logits"]

                # --- Filter out low-overlap ROIs ---
                filtered_pred_ranks, filtered_pred_class, filtered_pred_masks, filtered_indices, _ = filter_rois(img_mask, 
                                                                                        pred_masks,
                                                                                        pred_ranks,
                                                                                        pred_class,
                                                                                        device=device,
                                                                                        overlap_threshold=overlap_threshold,
                                                                                        confidence_threshold=confidence_threshold)



                # -------- Spearman Correlation (Ranking metric) --------
                metrics = calculate_metrics(gt_ranks, 
                                            pred_ranks, 
                                            filtered_pred_ranks, 
                                            masks, 
                                            filtered_indices=filtered_indices, 
                                            pred_masks=pred_masks, 
                                            gt_masks=gts)
                val_rho_sum += metrics["sor_sum"]
                n_rho += metrics["valid_sor_count"]
                valid_val_rhos += metrics["non_issue_rhos"]
                val_sasor_sum += metrics["sa_sor_sum"]
                n_sasor += metrics["valid_sa_sor_count"]
                val_mae_total += metrics["mae_sum"]
                n_mae += metrics["valid_mae_count"]
                
                # -------- Loss Computation --------
                losses_dict = compute_losses(gt_ranks, 
                   img_mask, 
                   gt_obj_class, 
                   pred_ranks,
                   pred_masks, 
                   pred_class,
                   valid_rois, 
                   filtered_pred_class, 
                   criterion_cls, 
                   device)
                rank_loss = losses_dict["rank_loss"]
                mask_loss = losses_dict["mask_loss"]
                class_loss = losses_dict["class_loss"]
                if isinstance(mask_loss, torch.Tensor):
                    mask_loss_val += mask_loss.detach().item()
                else:
                    mask_loss_val += float(mask_loss)
                if rank_loss == 0:
                    continue
                total_loss = rank_loss + mask_loss + class_loss
                val_loss_sum += total_loss.item()

                

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_mask_loss = mask_loss_val / len(val_loader)
        avg_val_sor = (1 + (val_rho_sum / n_rho if n_rho > 0 else 0)) / 2
        avg_val_sasor = val_sasor_sum / n_sasor if n_sasor > 0 else 0
        avg_val_mae = val_mae_total / n_mae if n_mae > 0 else 0


        print(f"Epoch [{epoch+1}/{num_epochs}] (Seed: {seed}) \n"
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | \n"
            f"Train Mask Loss: {avg_train_mask_loss:.4f} | Val Mask Loss: {avg_val_mask_loss:.4f} | \n"
            f"Train SOR (ρ): {avg_train_rho:.4f} ({valid_train_rhos}/{len(train_loader) * 4})| Val SOR (ρ): {avg_val_sor:.4f} ({valid_val_rhos}/{n_rho}) | \n"
            f"Train SA-SOR : {avg_train_sasor:.4f} | Val SA-SOR : {avg_val_sasor:.4f} | \n"
            f"Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f}")


        # print(f"Loss Breakdown:\n"
        #       f"\tPairwise Train: {avg_train_pairwise_loss:.4f} | Val: {avg_val_pairwise_loss:.4f}")

        # ---------------- MODEL SAVE ----------------
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_name = f"best_loss_run_{run_no}.pth"
            torch.save(model.state_dict(), os.path.join(model_save_path, best_model_name))
            print(f"✅ Saved new best model (val loss: {best_loss:.4f})")
        if avg_val_sor > best_sor:
            best_sor = avg_val_sor
            best_model_name = f"best_sor_run_{run_no}.pth"
            torch.save(model.state_dict(), os.path.join(model_save_path, best_model_name))
            print(f"✅ Saved new best model (val sor: {best_sor:.4f})")
        model_name = f"epoch_{epoch+1}_{run_no}_{str(avg_val_sor)[:6].replace('.', '-')}.pth"
        torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
        
    return best_sor

runs = 1
sors = []
for i in range(0,runs):
    best_sor = train(model_save_path=model_save_path, run_no=i, full_dataset=full_dataset, seed=42)
    sors.append(best_sor)
    print()

print(f"Best SOR score across {runs} runs: {sum(sors)/len(sors):.4f}")


