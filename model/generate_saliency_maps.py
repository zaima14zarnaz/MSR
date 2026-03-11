import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def create_sal_maps(gt_per_obj_masks, pred_per_obj_masks, ranks, max_objects=8):
    """
    pred_per_obj_masks: list of length B
        each entry is a tensor of shape (N_obj_b, H, W)
    ranks: tensor or list of per-image ranks, shape (B, K_rank)
    """

    B = len(pred_per_obj_masks)
    saliency_maps = []



    for i in range(B):

        masks_i = pred_per_obj_masks[i]   # (N_obj_i, H, W) or (0, H, W)
        ranks_i = ranks[i]

        # --------------------------------------
        # Convert ranks → NumPy
        # --------------------------------------
        if torch.is_tensor(ranks_i):
            ranks_i = ranks_i.detach().cpu().numpy()
        else:
            ranks_i = np.array(ranks_i)

        # --------------------------------------
        # If no predicted objects → blank map
        # --------------------------------------
        if masks_i.numel() == 0:
            # Must infer height/width from GT to avoid crash
            H = gt_per_obj_masks[i].shape[-2]
            W = gt_per_obj_masks[i].shape[-1]
            saliency_maps.append(torch.zeros((H, W)))
            print("Not generating maps")
            continue

        N_obj_i, H, W = masks_i.shape

        # --------------------------------------
        # Select usable count
        # --------------------------------------
        K_use = min(N_obj_i, len(ranks_i))

        if K_use == 0:
            saliency_maps.append(torch.zeros((H, W)))
            print("Not generating maps")
            continue

        # --------------------------------------
        # Compute normalized rank values
        # --------------------------------------
        pred_for_gt = ranks_i[:K_use]
        pred_order = np.argsort(np.argsort(pred_for_gt))
        normalized_ranks = (pred_order + 1) / K_use

        # --------------------------------------
        # Build the saliency canvas
        # --------------------------------------
        canvas = torch.zeros((H, W), dtype=torch.float32)

        for j in range(K_use):
            mask = masks_i[j]
            mask = (mask > 0)

            canvas[mask] = float(normalized_ranks[j])

        saliency_maps.append(canvas)

    return torch.stack(saliency_maps, dim=0)




def save_sal_map_gray(sal_map, path):
    if isinstance(sal_map, torch.Tensor):
        sal_map = sal_map.detach().cpu().numpy()

    sal_map = (sal_map * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(sal_map, mode="L").save(path)