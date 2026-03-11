import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error

def filter_rois(
    gt_masks,
    pred_masks,
    pred_ranks,
    pred_class,
    device,
    overlap_threshold=0.5,
    confidence_threshold=0.4,
    use_mask_max=False
):
    """
    Performs two-stage filtering:
    1. Overlap filtering   → keep only predictions matching GT masks
    2. Confidence filtering → final_conf = class_conf * mask_conf >= threshold

    Returns:
        filtered_pred_ranks
        filtered_pred_class
        filtered_pred_masks
        filtered_indices   (original pred mask indices)
        final_confidences  (optional debug)
    """

    filtered_pred_ranks = []
    filtered_pred_masks = []
    filtered_pred_class = []
    filtered_indices = []
    final_confidences = []

    # ============================================================
    # STEP 1 — Overlap filtering
    # ============================================================

    for i, pmask in enumerate(pred_masks):

        # Normalize shape (remove batch dim)
        if pmask.dim() == 3 and pmask.size(0) == 1:
            pmask = pmask.squeeze(0)

        pmask_bin = (pmask > 0.5).float()

        best_overlap = 0.0

        for gmask in gt_masks:

            # Resize predicted mask to GT size
            if pmask_bin.shape[-2:] != gmask.shape[-2:]:
                pmask_resized = F.interpolate(
                    pmask_bin[None, None],
                    size=gmask.shape[-2:],
                    mode="nearest"
                )[0, 0]
            else:
                pmask_resized = pmask_bin

            # Areas AFTER resizing
            pmask_area = pmask_resized.sum().item() + 1e-6
            gmask_area = gmask.sum().item() + 1e-6

            # Intersection
            inter = (pmask_resized * gmask).sum().item()

            # IOU-like match (IoM)
            overlap_ratio = inter / min(pmask_area, gmask_area)

            best_overlap = max(best_overlap, overlap_ratio)

        # Skip if overlap is too small
        if best_overlap < overlap_threshold:
            continue

        # TEMPORARILY add all overlap-valid predictions
        filtered_pred_ranks.append(pred_ranks[i])
        filtered_pred_masks.append(pred_masks[i])
        filtered_pred_class.append(pred_class[i])
        filtered_indices.append(i)

    # ============================================================
    # STEP 2 — Confidence filtering
    # ============================================================

    final_ranks = []
    final_masks = []
    final_classes = []
    final_indices = []

    for fp_idx, (r_score, c_logits, m_prob) in enumerate(
        zip(filtered_pred_ranks, filtered_pred_class, filtered_pred_masks)
    ):
        # ---- Class confidence ----
        if c_logits.numel() == 0:
            class_conf = torch.tensor(0.0, device=c_logits.device)
        elif c_logits.numel() > 1:
            class_conf = torch.softmax(c_logits, dim=-1)[0].max()
        else:
            class_conf = torch.sigmoid(c_logits)[0]


        # ---- Mask confidence ----
        if use_mask_max:
            mask_conf = m_prob.sigmoid().max()
        else:
            mask_conf = m_prob.sigmoid().mean()

        # ---- Final confidence ----
        final_conf = (class_conf * mask_conf).item()
        final_confidences.append(final_conf)

        if final_conf >= confidence_threshold:
            final_ranks.append(r_score)
            final_masks.append(m_prob)
            final_classes.append(c_logits)
            final_indices.append(filtered_indices[fp_idx])

    # ============================================================
    # FALLBACK — ensure the model returns something
    # ============================================================
    if len(final_ranks) == 0:
        return (
            filtered_pred_ranks,
            filtered_pred_class,
            filtered_pred_masks,
            filtered_indices,
            final_confidences
        )

    return (
        final_ranks,
        final_classes,
        final_masks,
        final_indices,
        final_confidences
    )


def calculate_metrics(gt_ranks, pred_ranks, filtered_pred_ranks, obj_masks_batch=None, pred_masks=None, gt_masks=None, seg_pred_masks=None, filtered_indices=None):

    # obj_masks_batch is a TUPLE of length BATCH SIZE
    # each element is tensor [N_obj_i, H, W]

    # Compute per-image, per-object sizes
    obj_sizes_batch = []
    if obj_masks_batch is not None:
        for obj_masks in obj_masks_batch:
            obj_sizes = compute_pixel_count_per_obj(obj_masks)   # returns list of sizes
            obj_sizes_batch.append(obj_sizes)

    # Proceed normally
    sor_sum, valid_sor_count, non_issue_rhos = calculate_sor(gt_ranks, pred_ranks, filtered_indices)
    
    if obj_masks_batch is not None:
        sa_sor_sum, valid_sa_sor_count, _ = calculate_sa_sor(
            gt_ranks=gt_ranks,
            pred_ranks=pred_ranks,
            obj_sizes_batch=obj_sizes_batch,
            filtered_indices=filtered_indices,
            obj_masks_batch=obj_masks_batch,
            pred_masks_batch=pred_masks
        )
        if seg_pred_masks is not None:
            mae_sum, valid_mae_count = calculate_mae(pred_masks_batch=seg_pred_masks, 
                                                    gt_masks_batch=gt_masks, 
                                                    pred_ranks_batch=pred_ranks, 
                                                    gt_ranks_batch=gt_ranks, 
                                                    filtered_indices_batch=filtered_indices)
        else:
            mae_sum, valid_mae_count = 0.0, 0

    else:
        sa_sor_sum, valid_sa_sor_count = 0.0, 0
        mae_sum, valid_mae_count = 0.0, 0

    return {
        "sor_sum": sor_sum,
        "valid_sor_count": valid_sor_count,
        "non_issue_rhos": non_issue_rhos,
        "sa_sor_sum": sa_sor_sum,
        "valid_sa_sor_count": valid_sa_sor_count,
        "mae_sum": mae_sum,
        "valid_mae_count": valid_mae_count
    }



def calculate_sor(gt_ranks, pred_ranks, filtered_indices):
    """
    gt_ranks: list of GT rank tensors (per image)
    pred_ranks: list of predicted rank tensors (per image)
    filtered_indices: list of kept ROI indices in the original prediction order
    """

    rho_sum = 0
    valid_sor_count = 0
    non_issue_rhos = 0

    for pred, gt in zip(pred_ranks, gt_ranks):

        # Convert predicted rank values and GT indices
        p = pred.detach().cpu().numpy().flatten()
        gt_idx = gt.cpu().numpy().astype(int)

        # --- Only include GT indices that exist in the prediction vector ---
        valid_idx = gt_idx[gt_idx < len(p)]

        if len(valid_idx) < 2:
            # print("<3")
            rho_sum += 1.0
            valid_sor_count += 1
            continue

        # --- Extract predicted scores at GT positions ---
        pred_for_gt = p[valid_idx]

        # --- Convert predicted scores to ranking order ---
        rank_pred = np.argsort(np.argsort(pred_for_gt))
        rank_gt   = np.arange(len(valid_idx))

        # ===============================================================
        # 🌟 Correct Filtering: KEEP ONLY GT POSITIONS IN filtered_indices
        # ===============================================================
        keep_mask = [idx in filtered_indices for idx in valid_idx]

        rank_pred = np.array(rank_pred)[keep_mask]
        rank_gt   = np.array(rank_gt)[keep_mask]

        # Handle cases where < 2 samples remain
        if len(rank_pred) < 2:
            # print("<3")
            rho_sum += 1.0
            valid_sor_count += 1
            continue

        # --- Compute Spearman ---
        rho, p_val = spearmanr(rank_pred, rank_gt)

        if not np.isnan(rho):
            rho_sum += rho
            valid_sor_count += 1
            non_issue_rhos += 1

    return rho_sum, valid_sor_count, non_issue_rhos


def calculate_sa_sor(gt_ranks, pred_ranks, filtered_indices, obj_sizes_batch,
                     pred_masks_batch, obj_masks_batch,
                     overlap_ratio_thresh=0.5):
    """
    SA-SOR with:
      - object-size weighting
      - filtered_indices masking
      - NEW: object excluded if predicted-mask overlap < 50% of GT object size
    """

    rho_sum = 0
    valid_sor_count = 0
    non_issue_rhos = 0

    for pred, gt, obj_sizes, pred_mask, obj_masks in zip(
        pred_ranks, gt_ranks, obj_sizes_batch, pred_masks_batch, obj_masks_batch):

        # Convert to numpy
        p = pred.detach().cpu().numpy().flatten()
        gt_idx = gt.cpu().numpy().astype(int)

        sizes = np.array(obj_sizes, dtype=float)

        # Fix predicted mask shape
        if pred_mask.dim() == 3 and pred_mask.size(0) == 1:
            pred_mask = pred_mask.squeeze(0)
        pred_mask_bin = (pred_mask > 0).cpu().numpy()  # [H, W]

        # Fix GT mask shape
        if obj_masks.dim() == 4 and obj_masks.size(1) == 1:
            obj_masks = obj_masks.squeeze(1)  # [K, H, W]

        obj_masks_np = (obj_masks > 0).cpu().numpy()

        # ------------------------------------------------------------
        # Step 1: validity = GT idx < pred length
        # ------------------------------------------------------------
        valid_idx = gt_idx[gt_idx < len(p)]

        # ------------------------------------------------------------
        # Step 2: NEW FILTER — remove objects with low mask overlap
        # ------------------------------------------------------------
        filtered_valid_idx = []
        for idx in valid_idx:
            if idx >= obj_masks_np.shape[0]:
                continue  # GT index has no corresponding mask → invalid

            gt_obj_mask = obj_masks_np[idx]     # [H, W]
            gt_size = sizes[idx]

            if gt_size <= 0:
                continue

            overlap = (pred_mask_bin & gt_obj_mask).sum()
            if overlap >= overlap_ratio_thresh * gt_size:
                filtered_valid_idx.append(idx)

        valid_idx = np.array(filtered_valid_idx, dtype=int)

        if len(valid_idx) < 2:
            rho_sum += 1.0
            valid_sor_count += 1
            # print("<2")
            continue

        # ------------------------------------------------------------
        # Extract predicted scores + sizes + ranks
        # ------------------------------------------------------------
        pred_for_gt = p[valid_idx]

        # Size alignment with fallback
        size_for_gt = []
        for idx in valid_idx:
            if idx < len(sizes):
                size_for_gt.append(sizes[idx])
            else:
                size_for_gt.append(0.0)
        size_for_gt = np.array(size_for_gt, dtype=float)

        # Convert pred scores → rank order
        rank_pred = np.argsort(np.argsort(pred_for_gt))
        rank_gt = np.arange(len(valid_idx))

        # ------------------------------------------------------------
        # Step 3: Apply your existing filtered_indices mask
        # ------------------------------------------------------------
        keep_mask = [idx in filtered_indices for idx in valid_idx]

        rank_pred = np.array(rank_pred)[keep_mask]
        rank_gt   = np.array(rank_gt)[keep_mask]
        weights   = np.array(size_for_gt)[keep_mask]

        if len(rank_pred) < 2:
            rho_sum += 1.0
            valid_sor_count += 1
            continue

        # ------------------------------------------------------------
        # Weighted Spearman (Pearson on weighted ranks)
        # ------------------------------------------------------------
        x = rank_pred.astype(float)
        y = rank_gt.astype(float)
        w = weights.astype(float)

        wx = np.average(x, weights=w)
        wy = np.average(y, weights=w)

        cov_xy = np.average((x - wx) * (y - wy), weights=w)
        var_x  = np.average((x - wx)**2, weights=w)
        var_y  = np.average((y - wy)**2, weights=w)

        if var_x == 0 or var_y == 0:
            rho = 1.0
        else:
            rho = cov_xy / np.sqrt(var_x * var_y)

        if not np.isnan(rho):
            rho_sum += rho
            valid_sor_count += 1
            non_issue_rhos += 1

    return rho_sum, valid_sor_count, non_issue_rhos




def compute_gt_centers(mask_tensor):
    """
    Computes (x, y) centers from a mask tensor that can be:
    - [N, H, W]
    - [B, N, H, W]
    """

    # If input is [B, N, H, W], flatten batch dimension
    if mask_tensor.dim() == 4:        # [B, N, H, W]
        B, N, H, W = mask_tensor.shape
        mask_tensor = mask_tensor.reshape(B * N, H, W)

    # If input is [N, H, W], do nothing

    if mask_tensor.dim() != 3:
        raise ValueError(f"Expected masks of shape [N,H,W] or [B,N,H,W], got {mask_tensor.shape}")

    centers = []

    for mask in mask_tensor:
        m = mask.detach().cpu().float()

        coords = torch.nonzero(m > 0.5, as_tuple=False)  # [k, 2]

        if coords.numel() == 0:
            centers.append([0.0, 0.0])
        else:
            # coords[:,0] = y indices, coords[:,1] = x indices
            y_mean = coords[:, 0].float().mean().item()
            x_mean = coords[:, 1].float().mean().item()
            centers.append([x_mean, y_mean])

    centers = np.array(centers, dtype=np.float32)
    return centers

import torch

def compute_pixel_count_per_obj(obj_masks):
    """
    obj_masks:
      - EITHER tensor [N, H, W] or [N, 1, H, W]  (one image)
      - OR list/tuple of such tensors            (batch of images)

    Returns:
      - if input is tensor:         [size_0, size_1, ..., size_{N-1}]
      - if input is list/tuple:     [[...sizes for img0...], [...for img1...], ...]
    """

    # Case 1: batch — list or tuple of per-image tensors
    if isinstance(obj_masks, (list, tuple)):
        return [compute_pixel_count_per_obj(m) for m in obj_masks]

    # Case 2: single tensor
    if not torch.is_tensor(obj_masks):
        raise TypeError(f"Expected tensor or list/tuple of tensors, got {type(obj_masks)}")

    m = obj_masks

    # Handle [N,1,H,W] → [N,H,W]
    if m.dim() == 4:
        # Most likely [N, 1, H, W], so squeeze channel dim
        if m.size(1) == 1:
            m = m.squeeze(1)   # [N, H, W]
        else:
            # Fallback: flatten leading dims to objects
            m = m.reshape(-1, m.size(-2), m.size(-1))

    if m.dim() != 3:
        raise ValueError(f"Expected [N,H,W] or [N,1,H,W], got {obj_masks.shape}")

    # Now m is [N, H, W]
    N = m.size(0)

    if N == 0:
        return []

    counts = (m > 0).reshape(N, -1).sum(dim=1)
    return counts.cpu().tolist()






def compute_gt_centers(rois):
    # rois: [N, 5] = [batch_idx, x1, y1, x2, y2]
    x1 = rois[:, 1]
    y1 = rois[:, 2]
    x2 = rois[:, 3]
    y2 = rois[:, 4]

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    return np.stack([xc.cpu().numpy(), yc.cpu().numpy()], axis=1)

def compute_pred_centers(pred_masks):
    """
    pred_masks is a LIST where each element is:
    - [H, W]
    - [1, H, W]
    - [C, H, W]
    - [1, 1, H, W]
    """

    centers = []
    # print(f"Pred masks length: {len(pred_masks)}")

    for m in pred_masks:

        # Squeeze all leading singleton dims
        # e.g. [1,1,H,W] → [H,W]
        while m.dim() > 2 and m.size(0) == 1:
            m = m.squeeze(0)

        # If shape is [C,H,W], reduce channels
        if m.dim() == 3:
            # Probability masks → take first channel
            m = m[0]

        if m.dim() != 2:
            raise RuntimeError(f"Unexpected pred mask shape: {m.shape}")

        m = m.detach().cpu().float()

        coords = torch.nonzero(m > 0.5, as_tuple=False)

        if coords.numel() == 0:
            centers.append([0.0, 0.0])
        else:
            y = coords[:, 0].float().mean().item()
            x = coords[:, 1].float().mean().item()
            centers.append([x, y])

    return np.array(centers, dtype=np.float32)

def calculate_mae(
    pred_masks_batch,
    gt_masks_batch,
    pred_ranks_batch,
    gt_ranks_batch,
    filtered_indices_batch
):

    mae_list = []

    for pred_ranks, gt_ranks, pred_mask, gt_masks, filtered_indices in zip(
        pred_ranks_batch, gt_ranks_batch, pred_masks_batch, gt_masks_batch, filtered_indices_batch
    ):

        # ----------------------------------------------------------
        # Normalize filtered_indices
        # ----------------------------------------------------------
        if isinstance(filtered_indices, int):
            filtered_indices = [filtered_indices]
        elif filtered_indices is None:
            filtered_indices = []
        elif isinstance(filtered_indices, np.ndarray):
            filtered_indices = filtered_indices.tolist()

        # ----------------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------------
        pred_scores = pred_ranks.detach().cpu().numpy().flatten()
        gt_idx_full = gt_ranks.cpu().numpy().astype(int)

        # Predicted mask → binary
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
            pred_mask_bin = (pred_mask.squeeze(0) > 0).cpu().numpy()
        else:
            pred_mask_bin = (pred_mask > 0).cpu().numpy()

        # GT masks → [K, H, W]
        if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
            gt_masks_np = (gt_masks.squeeze(1) > 0).cpu().numpy()
        else:
            gt_masks_np = (gt_masks > 0).cpu().numpy()

        K_gt, H, W = gt_masks_np.shape

        # ----------------------------------------------------------
        # Step 1: valid GT indices present in pred_scores
        # ----------------------------------------------------------
        valid_idx = [i for i in gt_idx_full if i < len(pred_scores)]

        # ----------------------------------------------------------
        # Step 2: apply filtered_indices
        # ----------------------------------------------------------
        final_idx = [i for i in valid_idx if i in filtered_indices]

        if len(final_idx) == 0:
            mae_list.append(0.0)
            continue

        # ----------------------------------------------------------
        # Build GT saliency map
        # ----------------------------------------------------------
        gt_map = np.zeros((H, W), dtype=np.float32)
        gt_rank_values = (np.arange(len(final_idx)) + 1) / len(final_idx)

        for j, idx in enumerate(final_idx):
            gt_map[gt_masks_np[idx] > 0] = gt_rank_values[j]

        # ----------------------------------------------------------
        # Build predicted rank map
        # ----------------------------------------------------------
        pred_for_gt = pred_scores[final_idx]
        pred_order = np.argsort(np.argsort(pred_for_gt))
        pred_rank_values = (pred_order + 1) / len(final_idx)

        pred_map = np.zeros((H, W), dtype=np.float32)
        for j, idx in enumerate(final_idx):
            pred_map[gt_masks_np[idx] > 0] = pred_rank_values[j]

        # ----------------------------------------------------------
        # Compute MAE
        # ----------------------------------------------------------
        mae = mean_absolute_error(gt_map.flatten(), pred_map.flatten())
        mae_list.append(mae)

    return sum(mae_list), len(mae_list)