import numpy as np
from skimage import measure
import torch

import numpy as np
from skimage import measure

def mask_to_coco_polygon(seg_mask_tensor, tolerance=0):
    mask = seg_mask_tensor.detach().cpu().numpy()

    # Debug statistics
    mask_max = mask.max()
    nonzero_count = np.sum(mask != 0)

    # print(
    #     f"mask_min={mask_min:.6f} "
    #     f"mask_max={mask_max:.6f} "
    #     f"sum={mask_sum:.2f} "
    #     f"zero_px={zero_count} "
    #     f"nonzero_px={nonzero_count}"
    # )

    # If mask has no valid signal → no polygons
    if nonzero_count == 0:
        return []

    # Dynamic threshold based on mask max
    threshold = 0.05 * mask_max

    # Binarize
    bin_mask = (mask > threshold).astype(np.uint8)

    # Pad for contour extraction
    padded = np.pad(bin_mask, pad_width=1, mode='constant', constant_values=0)

    # Extract contours
    contours = measure.find_contours(padded, 0.5)
    polygons = []

    for contour in contours:
        contour = contour - 1     # remove padding offset
        contour_xy = np.fliplr(contour)

        if tolerance > 0:
            contour_xy = measure.approximate_polygon(contour_xy, tolerance)

        if len(contour_xy) < 3:
            continue

        polygons.append(contour_xy.ravel().tolist())

    return polygons




import numpy as np
import cv2

def polygon_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask

def mask_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union


def batch_iou_masks(pred_masks, gt_masks):
    """
    pred_masks: (P, H, W) float tensor in {0,1}
    gt_masks:   (G, H, W) float tensor in {0,1}
    Returns:
        ious: (P, G) tensor of IoU values
    """
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        return torch.zeros((pred_masks.size(0), gt_masks.size(0)), device=pred_masks.device)

    # (P,1,H,W) & (1,G,H,W)
    p = pred_masks.unsqueeze(1)   # (P,1,H,W)
    g = gt_masks.unsqueeze(0)     # (1,G,H,W)

    inter = (p * g).sum(dim=(2, 3))  # (P,G)
    area_p = p.sum(dim=(2, 3))       # (P,1)
    area_g = g.sum(dim=(2, 3))       # (1,G)

    union = area_p + area_g - inter
    ious = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return ious

