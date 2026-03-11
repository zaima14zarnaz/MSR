import torch
import torch.nn as nn
import torch.nn.functional as F
def spearman_loss(pred, target):
    # Both: [n_obj]
    pred_rank = torch.argsort(torch.argsort(pred))
    target_rank = torch.argsort(torch.argsort(target))
    pred_rank = pred_rank.float()
    target_rank = target_rank.float()
    n = pred.numel()
    cov = ((pred_rank - pred_rank.mean()) * (target_rank - target_rank.mean())).mean()
    std = pred_rank.std() * target_rank.std() + 1e-6
    rho = cov / std
    return 1 - rho  # minimize 1 - rho → maximize correlation


def pairwise_ranking_loss(pred, target, margin=1.0):
    """
    Vectorized, stable pairwise margin ranking loss.
    Assumes higher target value = higher rank (more salient).
    """
    assert pred.ndim == 1 and target.ndim == 1, "Inputs must be 1D tensors [n_obj]"

    n = min(len(pred), len(target))
    if n < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred = pred[:n]
    target = target[:n]

    # Compute all pairwise differences
    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)      # [n, n]
    diff_target = target.unsqueeze(0) - target.unsqueeze(1)  # [n, n]

    # Binary comparison mask (exclude same targets)
    mask = diff_target != 0

    # Desired ordering: y_ij = sign(target_i - target_j)
    y = torch.sign(diff_target)
    y = torch.where(y == 0, torch.ones_like(y), y)  # ensure ±1 only

    # Margin ranking loss expects shape [N] vectors
    rank_loss_fn = nn.MarginRankingLoss(margin=margin, reduction='none')

    # Compute losses only for valid pairs
    valid_pred_diff = diff_pred[mask]
    valid_y = y[mask]

    if valid_pred_diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Apply margin ranking loss: wants y*(x1 - x2) > margin
    loss = rank_loss_fn(
        valid_pred_diff,
        torch.zeros_like(valid_pred_diff),
        valid_y
    )

    return loss.mean()

def listwise_ranking_loss(pred, target):
    """
    Listwise ranking loss (ListNet formulation).
    Enforces global ordering consistency between predicted and target ranks.

    Args:
        pred (Tensor): [n_obj] predicted scores.
        target (Tensor): [n_obj] ground-truth ranks (higher means more salient).
    """
    assert pred.ndim == 1 and target.ndim == 1, "Inputs must be 1D tensors [n_obj]"

    n = min(len(pred), len(target))
    if n < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred, target = pred[:n], target[:n]

    # Convert ranks to probability distributions
    p = F.log_softmax(pred, dim=0)      # predicted log-probs
    q = F.softmax(-target, dim=0)       # target probs (invert rank order)

    # Cross entropy between q and p
    loss = -(q * p).sum()
    return loss


def ranknet_loss(rank_scores, gt_ranks):
    i_idx, j_idx = torch.triu_indices(len(rank_scores), len(rank_scores), offset=1)
    s_i, s_j = rank_scores[i_idx], rank_scores[j_idx]
    gt_i, gt_j = gt_ranks[i_idx], gt_ranks[j_idx]

    y_ij = (gt_i > gt_j).float()
    diff = s_i - s_j
    return torch.nn.functional.binary_cross_entropy_with_logits(diff, y_ij)

def listnet_loss(scores, ranks, mask=None):
    # scores: [N_b] predicted, ranks: [N_b] lower=less salient (or normalize to [0..1])
    # build soft targets by exp(-rank)
    tgt = torch.softmax(-ranks.float(), dim=0)
    prob = torch.log_softmax(scores, dim=0)
    if mask is not None: tgt, prob = tgt[mask], prob[mask]
    return -(tgt * prob).sum()


import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_listnet(pred, target, margin=1.0, alpha=0.2):
    """
    Combined Pairwise + ListNet loss.
    Pairwise: Margin ranking loss based on relative order.
    ListNet: Listwise loss based on probability distributions over ranks.
    """
    assert pred.ndim == 1 and target.ndim == 1, "Inputs must be 1D tensors [n_obj]"

    n = min(len(pred), len(target))
    if n < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred = pred[:n]
    target = target[:n]

    # ===== Pairwise Margin Ranking Loss =====
    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)      # [n, n]
    diff_target = target.unsqueeze(0) - target.unsqueeze(1)  # [n, n]

    mask = diff_target != 0
    y = torch.sign(diff_target)
    y = torch.where(y == 0, torch.ones_like(y), y)  # ensure ±1 only

    rank_loss_fn = nn.MarginRankingLoss(margin=margin, reduction='none')
    valid_pred_diff = diff_pred[mask]
    valid_y = y[mask]

    if valid_pred_diff.numel() == 0:
        pairwise_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
    else:
        pairwise_loss = rank_loss_fn(valid_pred_diff,
                                     torch.zeros_like(valid_pred_diff),
                                     valid_y).mean()

    # ===== ListNet Loss (Listwise) =====
    # Convert scores to probability distributions using softmax
    pred_prob = F.softmax(pred, dim=0)
    target_prob = F.softmax(target, dim=0)

    # Cross entropy between predicted and target distributions
    listnet_loss = -torch.sum(target_prob * torch.log(pred_prob + 1e-8))

    # ===== Combined Loss =====
    total_loss = pairwise_loss + alpha * listnet_loss
    return total_loss


def soft_spearman_loss(pred, target, eps=1e-8):
    """
    Differentiable approximation of (1 - Spearman correlation) / 2.
    Computes soft rank correlation between pred and target.
    """
    pred_rank = torch.argsort(torch.argsort(pred))
    target_rank = torch.argsort(torch.argsort(target))

    # Convert to float and normalize ranks
    pred_rank = pred_rank.float()
    target_rank = target_rank.float()

    pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + eps)
    target_rank = (target_rank - target_rank.mean()) / (target_rank.std() + eps)

    # Spearman correlation ≈ Pearson correlation of ranks
    spearman_corr = torch.sum(pred_rank * target_rank) / (len(pred_rank) - 1)
    spearman_loss = 1.0 - spearman_corr  # convert to loss (maximize corr → minimize 1 - corr)
    return spearman_loss

def dice_loss(pred, target, smooth=1e-6):

    if isinstance(pred, list):
        pred = [p for p in pred if p is not None and p.numel() > 0]
        if len(pred) == 0:
            return torch.tensor(0.0, device=target.device)
        pred = torch.cat(pred, dim=0)

    if isinstance(target, list):
        target = [t for t in target if t is not None and t.numel() > 0]
        if len(target) == 0:
            return torch.tensor(0.0, device=pred.device)
        target = torch.cat(target, dim=0)

    # ---- Fix missing spatial dims ----
    if target.dim() == 2:  # (B, H*W) or (B, H)
        L = target.size(1)
        S = int(L ** 0.5)
        if S * S == L:  # square
            target = target.view(target.size(0), 1, S, S)
        else:
            raise ValueError(f"Target mask is 1D and non-square: {target.shape}")

    if pred.dim() == 2:
        L = pred.size(1)
        S = int(L ** 0.5)
        if S * S == L:
            pred = pred.view(pred.size(0), 1, S, S)
        else:
            raise ValueError(f"Pred mask is 1D and non-square: {pred.shape}")

    # ---- Ensure channel dim ----
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    # ---- Resize ----
    if pred.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")

    # ---- Align batch ----
    min_batch = min(pred.shape[0], target.shape[0])
    pred = pred[:min_batch]
    target = target[:min_batch]

    # ---- Flatten ----
    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (pred * target).sum(dim=1)
    dice = (2 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1 - dice.mean()


def class_ce_loss(gt_class, pred_class, criterion_cls, device):
    if pred_class is None:
        return torch.tensor(0.0, device=device)

    if isinstance(pred_class, list):
        pred_class = [p for p in pred_class if p is not None and p.numel() > 0]
        if len(pred_class) == 0:
            return torch.tensor(0.0, device=device)
        pred_class = torch.cat(pred_class, dim=0)

    if pred_class.numel() == 0:
        return torch.tensor(0.0, device=device)

    pred_class = pred_class.float()
    gt_class = gt_class.to(pred_class.device).long()

    min_len = min(pred_class.size(0), gt_class.size(0))
    pred_class = pred_class[:min_len]
    gt_class = gt_class[:min_len]

    return criterion_cls(pred_class, gt_class)



import torch
import torch.nn.functional as F

def calculate_prompt_loss(roi_prompts, ranks, same_scale=1.0, diff_scale=1.0):
    """
    roi_prompts: list of [n_rois_b, D] tensors (ROI embeddings per image)
    ranks: list of [n_rois_b] tensors (rank indices per image)
    - Each rank tensor defines the rank order, NOT direct rank labels.
      e.g., tensor([3,2,4,0]) means:
        roi[3] -> rank1 (highest),
        roi[2] -> rank2,
        roi[4] -> rank3,
        roi[0] -> rank4.
    - Goal:
        * Same-rank ROIs across images → small MSE (similar)
        * Different-rank ROIs across images → large MSE (dissimilar)
    """

    device = next(iter(roi_prompts)).device
    D = roi_prompts[0].size(-1)

    # --- Build mapping rank -> roi index for each image ---
    rank_to_roi = []
    for roi_vecs, rank_tensor in zip(roi_prompts, ranks):
        mapping = {}
        n_ranked = len(rank_tensor)
        for rank, roi_idx in enumerate(rank_tensor):
            mapping[rank + 1] = roi_idx.item()  # rank1 = highest
        rank_to_roi.append(mapping)

    all_ranks = sorted(set(r for m in rank_to_roi for r in m.keys()))
    total_loss, total_pairs = 0.0, 0

    # --- SAME-RANK MSE LOSS (want low MSE) ---
    for r in all_ranks:
        rois_r = []
        for b, mapping in enumerate(rank_to_roi):
            if r in mapping:
                roi_idx = mapping[r]
                rois_r.append(roi_prompts[b][roi_idx])
        if len(rois_r) < 2:
            continue  # need at least 2 images with same rank

        rois_r = torch.stack(rois_r)  # [num_imgs_with_rank_r, D]
        # pairwise mse between all roi pairs of same rank
        diff = rois_r.unsqueeze(1) - rois_r.unsqueeze(0)  # [N, N, D]
        mse = (diff ** 2).mean(dim=-1)  # [N, N]
        eye = torch.eye(mse.size(0), device=device)
        same_loss = mse * (1 - eye)
        total_loss += same_scale * same_loss.sum() / (same_loss.numel() - mse.size(0))
        total_pairs += 1

    # --- DIFFERENT-RANK MSE LOSS (want high MSE) ---
    for r1 in all_ranks:
        for r2 in all_ranks:
            if r1 >= r2:
                continue
            rois_r1, rois_r2 = [], []
            for b, mapping in enumerate(rank_to_roi):
                if r1 in mapping and r2 in mapping:
                    roi_idx1 = mapping[r1]
                    roi_idx2 = mapping[r2]
                    rois_r1.append(roi_prompts[b][roi_idx1])
                    rois_r2.append(roi_prompts[b][roi_idx2])
            if len(rois_r1) < 1 or len(rois_r2) < 1:
                continue

            rois_r1 = torch.stack(rois_r1)
            rois_r2 = torch.stack(rois_r2)
            # pairwise mse between all roi pairs of different ranks
            diff = rois_r1.unsqueeze(1) - rois_r2.unsqueeze(0)  # [N1, N2, D]
            mse = (diff ** 2).mean(dim=-1)
            # maximize dissimilarity -> minimize (margin - mse)
            diff_loss = F.relu(1.0 - mse)  # higher mse -> lower loss
            total_loss += diff_scale * diff_loss.mean()
            total_pairs += 1

    if total_pairs == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / total_pairs

def compute_losses(gt_ranks, 
                   gt_masks, 
                   gt_class, 
                   pred_ranks,
                   pred_masks, 
                   pred_class,
                   valid_rois, 
                   filtered_pred_class, 
                   criterion_cls, 
                   device):
    # ----- Restructure tensors for loss computation -------
    if isinstance(pred_ranks, list):
        pred_ranks = torch.cat(pred_ranks, dim=0)
    if isinstance(gt_ranks, list):
        gt_r = torch.cat(gt_ranks, dim=0)
    else:
        gt_r = gt_ranks
    gt_r = gt_r.to(pred_ranks.device).float()

    if len(valid_rois) == 0 or len(filtered_pred_class) == 0:
        # Skip classification loss entirely for empty ROI batches
        class_loss = torch.tensor(0.0, device=device)
        rank_loss = torch.tensor(0.0, device=device)
        mask_loss = torch.tensor(0.0, device=device)
        total_loss = class_loss
        return {
            "rank_loss": 0, 
            "mask_loss": 0,
            "class_loss": class_loss
        }




    # Compute ranking and mask losses
    mask_loss = dice_loss(pred=pred_masks, target=gt_masks)
    class_loss = class_ce_loss(gt_class, pred_class, criterion_cls, device)
    rank_loss = pairwise_listnet(pred_ranks.view(-1), gt_r.view(-1))

    return {
        "rank_loss": rank_loss, 
        "mask_loss": mask_loss,
        "class_loss": class_loss
    }




