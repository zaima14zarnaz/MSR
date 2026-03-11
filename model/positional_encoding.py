import math
import torch
import torch.nn as nn

import math
import torch

def build_2d_sincos_pos_embed(positions, embed_dim, img_size=None):
    """
    Compute 2D sinusoidal positional encodings for ROI centers and sizes.
    --------------------------------------------------------------------
    positions: [N, 4] tensor of (x_center, y_center, width, height) in pixels.
    embed_dim: total embedding dimension (must be divisible by 8).
    img_size:  (H, W) tuple or tensor, required for normalization if ROIs are not normalized.
    Returns: [N, embed_dim] positional encodings
    """
    device = positions.device
    N, D = positions.shape  # [N, 4]
    assert D == 4, "positions must be (x_center, y_center, width, height)"

    # ---- Normalize coordinates to [0, 1] ----
    if img_size is not None:
        if isinstance(img_size, (tuple, list)):
            H, W = img_size
        elif torch.is_tensor(img_size):
            H, W = img_size[0].item(), img_size[1].item()
        else:
            raise TypeError("img_size must be a (H, W) tuple or a tensor")

        norm = torch.tensor([W, H, W, H], device=device, dtype=positions.dtype)
        positions = positions / (norm + 1e-6)
    else:
        # fallback normalization (if image size not provided)
        positions = positions / (positions.max(dim=0, keepdim=True)[0] + 1e-6)

    # ---- Compute 2D sinusoidal embeddings ----
    assert embed_dim % (2 * D) == 0, f"embed_dim ({embed_dim}) must be divisible by {2 * D}"
    div_term = torch.exp(
        torch.arange(0, embed_dim // (2 * D), device=device) * -(math.log(10000.0) / (embed_dim // (2 * D)))
    )

    pe_list = []
    for i in range(D):
        pos = positions[:, i].unsqueeze(1)  # [N, 1]
        pe_sin = torch.sin(pos * div_term)
        pe_cos = torch.cos(pos * div_term)
        pe_list.extend([pe_sin, pe_cos])

    pos_embed = torch.cat(pe_list, dim=1)  # [N, embed_dim]
    return pos_embed


def roi_to_xywh(rois):
    """
    Convert ROI tensor from [batch_idx, x1, y1, x2, y2]
    → [x_center, y_center, width, height]
    """
    _, x1, y1, x2, y2 = rois.unbind(dim=1)
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2
    y_c = y1 + h / 2
    return torch.stack([x_c, y_c, w, h], dim=1)
