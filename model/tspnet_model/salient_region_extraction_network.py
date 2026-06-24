"""
Criteria: Backbones
Variant No: 1A
Variant Focus: Resnet 101
Remaining Pipeline Description: Best configuration from A-I
SOR: 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import clip
import math

# --------------------------
# Deformable alignment block
# --------------------------
try:
    from mmcv.ops import DeformConv2d  # optional
    _HAS_MMCV = True
except Exception:
    DeformConv2d = None
    _HAS_MMCV = False


class DeformAlignBlock(nn.Module):
    """
    Lightweight per-scale alignment/refinement block.
    If MMCV's DeformConv2d is available, uses learned offsets.
    Otherwise, falls back to a 3x3 Conv2d as a safe approximation.
    """
    def __init__(self, channels: int, k: int = 3, groups: int = 1):
        super().__init__()
        self.use_deform = _HAS_MMCV
        if self.use_deform:
            # offset conv: 2*k*k channels for (dx, dy) per kernel location
            self.offset_conv = nn.Conv2d(channels, 2 * k * k, kernel_size=3, padding=1)
            self.deform = DeformConv2d(channels, channels, kernel_size=k, padding=k // 2, groups=groups)
        else:
            self.conv = nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=groups, bias=False)

        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        if self.use_deform:
            offset = self.offset_conv(x)
            x = self.deform(x, offset)
        else:
            x = self.conv(x)
        return self.act(self.norm(x))

class BSDHeadMAF(nn.Module):
    def __init__(self, in_channels, out_channels=256, pool_size=7, dropout_p=0.2):
        super().__init__()
        self.pool_size = pool_size
        self.out_channels = out_channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.maf = MultiKernelFusion(out_channels, dropout=dropout_p, phrase_dim=512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )

    @staticmethod
    def _spatial_scale(feature_map, image_shape):
        Hf, Wf = feature_map.shape[-2:]
        Himg, Wimg = image_shape
        sf_h = Hf / float(Himg)
        sf_w = Wf / float(Wimg)
        return sf_w  # uniform scaling in typical backbones

    def _roi_align_and_project(self, feat_map, rois, image_shape, phrase_context=None):
        # --- Ensure rois is always [N, 5] (batch_idx, x1, y1, x2, y2) ---
        if isinstance(rois, list):
            if len(rois) == 0 or all(r.numel() == 0 for r in rois):
                rois = feat_map.new_zeros((0, 5))
            else:
                rois = torch.cat([r for r in rois if r.numel() > 0], dim=0)
        elif torch.is_tensor(rois):
            if rois.numel() == 0 or rois.dim() != 2 or rois.size(-1) != 5:
                rois = feat_map.new_zeros((0, 5))
        else:
            print(rois)
            raise TypeError(f"Invalid rois type: {type(rois)}")
        rois = rois.to(feat_map.device, dtype=torch.float32).contiguous()
        if rois.numel() == 0:
            print(rois)
            return feat_map.new_zeros((0, self.out_channels, self.pool_size, self.pool_size))

        s = self._spatial_scale(feat_map, image_shape)
        pooled = roi_align(
            feat_map, rois,
            output_size=self.pool_size,
            spatial_scale=s,
            sampling_ratio=-1
        )
        x = self.proj(pooled)   # [N_rois, C, P, P]
        # phrase_context must be [N_rois, 512] and row-aligned with ROI features.
        if (
            phrase_context is None
            or (not torch.is_tensor(phrase_context))
            or phrase_context.numel() == 0
            or phrase_context.dim() != 2
            or phrase_context.size(0) != x.size(0)
        ):
            phrase_context = None
        else:
            phrase_context = phrase_context.to(device=x.device)

        x = self.maf(x, phrase_context)  # [N_rois, C, P, P]
        return x

    # NEW: expose pre-GAP map
    def forward_map(self, feat_map, rois, image_shape, phrase_context=None):
        return self._roi_align_and_project(feat_map, rois, image_shape, phrase_context=phrase_context)

    # # Original behavior (kept)
    # def forward(self, feat_map, rois, image_shape):
    #     x = self._roi_align_and_project(feat_map, rois, image_shape)
    #     if x.size(0) == 0:
    #         return feat_map.new_zeros((0, self.out_channels))
    #     return self.head(x)
    
class PhraseDynamicDepthwiseConv(nn.Module):
    def __init__(self, channels, phrase_dim=512, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.phrase_dim = phrase_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = dilation * (kernel_size // 2)
        self.kernel_elems = kernel_size * kernel_size

        self.base_kernel = nn.Parameter(torch.zeros(1, channels, self.kernel_elems))
        nn.init.kaiming_uniform_(self.base_kernel, a=math.sqrt(5))
        self.raw_token_proj = nn.Linear(77, phrase_dim)
        self.delta_kernel_proj = nn.Linear(phrase_dim, channels * self.kernel_elems)
        self.kernel_scale = nn.Parameter(torch.tensor(0.01))
        self.phrase_norm = nn.LayerNorm(phrase_dim)
        self.phrase_strength = nn.Parameter(torch.tensor(-3.0))

        self.dropout2d = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x, phrase_vec=None):
        n, c, h, w = x.shape

        if phrase_vec is None:
            phrase_feat = x.new_zeros((n, self.phrase_dim))
        else:
            phrase_vec = phrase_vec.to(device=x.device)
            if phrase_vec.dim() != 2:
                phrase_vec = phrase_vec.view(phrase_vec.size(0), -1)
            if phrase_vec.size(0) == 1 and n > 1:
                phrase_vec = phrase_vec.expand(n, -1)
            elif phrase_vec.size(0) != n:
                phrase_feat = x.new_zeros((n, self.phrase_dim))
            elif phrase_vec.size(1) == self.phrase_dim:
                phrase_feat = phrase_vec.to(dtype=x.dtype)
            else:
                phrase_tokens = phrase_vec.to(dtype=x.dtype)
                if phrase_tokens.size(1) > 77:
                    phrase_tokens = phrase_tokens[:, :77]
                else:
                    pad = x.new_zeros((n, 77 - phrase_tokens.size(1)))
                    phrase_tokens = torch.cat([phrase_tokens, pad], dim=1)
                phrase_feat = self.raw_token_proj(phrase_tokens)

        phrase_feat = self.phrase_norm(phrase_feat)

        delta_kernel = torch.tanh(self.delta_kernel_proj(phrase_feat)).view(n, c, self.kernel_elems)
        kernel = self.base_kernel.to(dtype=x.dtype) + self.kernel_scale.to(dtype=x.dtype) * delta_kernel

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=1,
        )
        patches = patches.view(n, c, self.kernel_elems, h * w)
        out = (patches * kernel.unsqueeze(-1)).sum(dim=2).view(n, c, h, w)

        out = self.dropout2d(out)
        out = self.post(out)
        alpha = torch.sigmoid(self.phrase_strength).to(dtype=x.dtype)
        return x + alpha * out


# ----------------------------------------------
# MAFormer-style local/global fusion (per ROI)
# ----------------------------------------------
class MultiKernelFusion(nn.Module):
    """
    MultiKernelFusion (multi-branch MAFormer-style fusion per ROI)
    -------------------------------------------
    - Builds 3 local path features (no separate global kernel branch)
    - Each spatial location has a 3-token sequence (one per path)
    - Transformer Encoder (4 layers) aggregates cross-path relationships
    - Multi-head self-attention fuses the P path tokens (dense; avoids PyG sparse GAT on huge batched graphs)
    - Optional downsampling (token_stride) for efficiency
    """

    def __init__(self, channels: int, dropout: float = 0.0,
                 num_heads: int = 4, token_stride: int = 1, phrase_dim: int = 512):
        super().__init__()
        self.channels = channels
        self.token_stride = token_stride
        self.phrase_dim = phrase_dim
        self.num_paths = 3

        # ===== Local feature branches =====
        self.local_k1 = PhraseDynamicDepthwiseConv(
            channels=channels, phrase_dim=phrase_dim, kernel_size=3, dilation=1, dropout=dropout
        )
        self.local_k2 = PhraseDynamicDepthwiseConv(
            channels=channels, phrase_dim=phrase_dim, kernel_size=5, dilation=1, dropout=dropout
        )
        self.local_k3 = PhraseDynamicDepthwiseConv(
            channels=channels, phrase_dim=phrase_dim, kernel_size=3, dilation=2, dropout=dropout
        )

        # ===== Optional spatial downsampling =====
        if token_stride > 1:
            self.shrink = nn.AvgPool2d(kernel_size=token_stride, stride=token_stride)
        else:
            self.shrink = nn.Identity()

        # ===== Transformer Fusion =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads,
            dim_feedforward=channels,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # ===== Path-token fusion (all-to-all over P=3 tokens per spatial cell) =====
        self.path_attn_fuse = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ===== Output projection =====
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x, phrase_vec=None):
        N, C, H, W = x.shape

        branch_phrase_vec = None
        if phrase_vec is not None:
            phrase_vec = phrase_vec.to(device=x.device)
            if phrase_vec.dim() != 2:
                phrase_vec = phrase_vec.view(phrase_vec.size(0), -1)
            if phrase_vec.size(0) == 1 and N > 1:
                phrase_vec = phrase_vec.expand(N, -1)
            elif phrase_vec.size(0) != N:
                phrase_vec = None
            branch_phrase_vec = phrase_vec

        # ----- Local paths -----
        x_l1 = self.local_k1(x, phrase_vec=branch_phrase_vec)
        x_l2 = self.local_k2(x, phrase_vec=branch_phrase_vec)
        x_l3 = self.local_k3(x, phrase_vec=branch_phrase_vec)

        # stack paths: [N, P, C, H, W]
        all_feats = torch.stack([x_l1, x_l2, x_l3], dim=1)
        P = self.num_paths

        # optional downsampling
        if self.token_stride > 1:
            all_feats = all_feats.view(N * P, C, H, W)
            all_feats = self.shrink(all_feats)
            Hs, Ws = all_feats.shape[-2:]
            all_feats = all_feats.view(N, P, C, Hs, Ws)
        else:
            Hs, Ws = H, W

        # reshape for transformer: each (H,W) location has P path tokens
        feats = all_feats.permute(0, 3, 4, 1, 2).contiguous()  # [N, Hs, Ws, P, C]
        G = N * Hs * Ws
        tokens = feats.view(G, P, C)

        fused_tokens = self.transformer_fusion(tokens)  # [G, P, C]

        attn_out, _ = self.path_attn_fuse(
            fused_tokens, fused_tokens, fused_tokens, need_weights=False
        )
        out = attn_out.mean(dim=1)  # [G, C]

        # reshape back to image-like format
        fused = out.view(N, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
        if (Hs, Ws) != (H, W):
            fused = F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=False)

        return self.out_proj(fused) + x


def _debug_test_phrase_conditioned_maf():
    """
    Temporary debug check for phrase-conditioned MultiKernelFusion.
    Kept as a helper and not called by default.
    """
    if not torch.cuda.is_available():
        print("[DEBUG][MAF] CUDA not available, skipping test.")
        return

    x = torch.randn(6, 256, 7, 7).cuda()
    phrase_a = torch.randn(6, 512).cuda()
    phrase_b = torch.randn(6, 512).cuda()

    maf = MultiKernelFusion(256, phrase_dim=512, dropout=0.2).cuda()
    maf.eval()
    with torch.no_grad():
        y_a = maf(x, phrase_a)
        y_b = maf(x, phrase_b)
        y_none = maf(x, None)

    print("[DEBUG][MAF] y_a.shape:", y_a.shape)
    print("[DEBUG][MAF] max|y_a - y_b|:", (y_a - y_b).abs().max().item())
    print("[DEBUG][MAF] max|y_a - y_none|:", (y_a - y_none).abs().max().item())

    assert y_a.shape == x.shape
    assert torch.isfinite(y_a).all()
    assert torch.isfinite(y_b).all()
    assert torch.isfinite(y_none).all()
    print("[DEBUG][MAF] passed.")

# Temporary manual call (keep disabled after verification):
# _debug_test_phrase_conditioned_maf()


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossScaleAttentionFusion(nn.Module):
    """
    Cross-Scale Attention Fusion v2 (Dual Cross-Attention)
    ------------------------------------------------------
    - Treats multi-scale ROI maps as scale tokens
    - Two layers of cross-attention + FFN fusion
    - Each layer refines inter-scale dependencies
    - Learnable positional embeddings for scale order
    """
    def __init__(self, channels: int, num_scales: int = 4, num_heads: int = 4, dropout: float = 0.1, phrase_dim: int = 512):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.phrase_dim = phrase_dim

        # 1️⃣ Projection to reduce computational cost before attention
        self.scale_proj = nn.Linear(channels, channels)
        self.pos_embed = nn.Parameter(torch.randn(1, num_scales, channels))

        # 2️⃣ Two stacked cross-attention layers
        self.cross_attn1 = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward networks for both layers
        self.ffn1 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )

        # LayerNorms (pre-norm style)
        self.norm1a = nn.LayerNorm(channels)
        self.norm1b = nn.LayerNorm(channels)
        self.norm2a = nn.LayerNorm(channels)
        self.norm2b = nn.LayerNorm(channels)

        self.phrase_to_scales = nn.Sequential(
            nn.Linear(phrase_dim, channels * num_scales),
            nn.LayerNorm(channels * num_scales),
            nn.GELU()
        )
        self.spatial_key_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # ------------------------------------------------------------------
        # Raw-phrase token -> spatial feature attention (replaces prompt_mask/att_mask)
        # ------------------------------------------------------------------
        # raw_phrase_tokens are CLIP token IDs (default CLIP context length 77)
        self.raw_token_embed = nn.Embedding(49408, channels, padding_idx=0)
        self.raw_token_pos_embed = nn.Embedding(77, channels)
        self.raw_phrase_norm = nn.LayerNorm(channels)
        self.raw_phrase_ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
        )

        self.phrase_to_spatial_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_scales)
        ])

        self.spatial_to_phrase_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_scales)
        ])

        self.phrase_context_proj = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1)
            for _ in range(num_scales)
        ])

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.dropout2d = nn.Dropout2d(dropout)

    def _embed_raw_phrase_tokens(self, raw_phrase_tokens, device):
        raw_phrase_tokens = raw_phrase_tokens.to(device=device, dtype=torch.long)

        N, L = raw_phrase_tokens.shape

        pad_mask = raw_phrase_tokens == 0
        valid_rows = (raw_phrase_tokens != 0).any(dim=1)

        token_emb = self.raw_token_embed(raw_phrase_tokens)

        pos_ids = torch.arange(
            L,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(N, L)

        token_emb = token_emb + self.raw_token_pos_embed(pos_ids)
        token_emb = self.raw_phrase_norm(token_emb)
        token_emb = self.raw_phrase_ffn(token_emb)

        token_emb = torch.nan_to_num(
            token_emb,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        return token_emb, pad_mask, valid_rows

    def _apply_raw_phrase_spatial_attention(
        self,
        feat,
        raw_phrase_tokens,
        scale_id,
    ):
        # feat: [N, C, H, W], raw_phrase_tokens: [N, 77]
        _, C, H, W = feat.shape
        dtype = feat.dtype

        if raw_phrase_tokens is None:
            return feat

        phrase_tokens, pad_mask, valid_rows = self._embed_raw_phrase_tokens(
            raw_phrase_tokens,
            device=feat.device,
        )

        if not valid_rows.any():
            return feat

        feat_out = feat.clone()

        feat_valid = feat[valid_rows]            # [Nv, C, H, W]
        phrase_valid = phrase_tokens[valid_rows]  # [Nv, 77, C]
        pad_valid = pad_mask[valid_rows]        # [Nv, 77]

        # Spatial tokens: [Nv, H*W, C]
        spatial_tokens = feat_valid.flatten(2).transpose(1, 2).to(dtype=phrase_valid.dtype)

        # Phrase tokens attend spatial tokens: weights [Nv, 77, H*W]
        _, phrase_to_spatial_weights = self.phrase_to_spatial_attn[scale_id](
            query=phrase_valid,
            key=spatial_tokens,
            value=spatial_tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        token_valid = (~pad_valid).to(dtype=phrase_to_spatial_weights.dtype).unsqueeze(-1)

        # Aggregate to a spatial attention map in [0, 1]
        spatial_attn = (
            phrase_to_spatial_weights * token_valid
        ).sum(dim=1) / token_valid.sum(dim=1).clamp(min=1.0)  # [Nv, H*W]

        spatial_attn = spatial_attn.view(feat_valid.size(0), 1, H, W).clamp(0.0, 1.0)

        # Spatial tokens attend back to phrase tokens to create a phrase context map
        phrase_context_tokens, _ = self.spatial_to_phrase_attn[scale_id](
            query=spatial_tokens,
            key=phrase_valid,
            value=phrase_valid,
            key_padding_mask=pad_valid,
            need_weights=False,
        )  # [Nv, H*W, C]

        phrase_context_map = phrase_context_tokens.transpose(1, 2).view(
            feat_valid.size(0),
            C,
            H,
            W,
        )

        phrase_context_map = self.phrase_context_proj[scale_id](
            phrase_context_map.to(dtype=dtype)
        )


        updated_valid = feat_valid + spatial_attn.to(dtype=dtype) * phrase_context_map

        feat_out[valid_rows] = updated_valid

        return feat_out

    def forward(self, roi_maps_list, raw_phrase_tokens=None, phrase_context=None, att_mask=None):
        # raw_phrase_tokens are matched raw CLIP token IDs per ROI [N_rois, 77].
        # Phrase token IDs directly attend to spatial feature-map tokens at each scale.
        # This replaces the previous prompt_mask/att_mask projection as the conditioning path.
        # CLIP is not used inside this fusion module; it only consumes token IDs.
        if len(roi_maps_list) == 0:
            raise ValueError("roi_maps_list is empty")

        N, C, H, W = roi_maps_list[0].shape
        num_scales = len(roi_maps_list)

        if raw_phrase_tokens is not None:
            conditioned_maps = []
            for s, feat in enumerate(roi_maps_list):
                if s < self.num_scales:
                    feat = self._apply_raw_phrase_spatial_attention(
                        feat,
                        raw_phrase_tokens,
                        s,
                    )
                conditioned_maps.append(feat)
            roi_maps_list = conditioned_maps
        else:
            # Legacy fallback: condition using phrase embeddings.
            # att_mask is intentionally ignored here (legacy prompt_mask projection removed elsewhere).
            use_phrase_conditioning = (
                phrase_context is not None
                and phrase_context.numel() > 0
                and phrase_context.dim() == 2
                and phrase_context.shape[0] == N
                and phrase_context.shape[1] == self.phrase_dim
                and num_scales == self.num_scales
            )
            if use_phrase_conditioning:
                scale_queries = self.phrase_to_scales(phrase_context)
                scale_queries = scale_queries.view(N, num_scales, C)
                conditioned_maps = []
                for s in range(num_scales):
                    feat = roi_maps_list[s]
                    q = scale_queries[:, s, :]
                    k = self.spatial_key_proj(feat)
                    q = F.normalize(q, dim=-1)
                    k = F.normalize(k, dim=1)
                    logits = torch.einsum("nc,nchw->nhw", q, k) / math.sqrt(C)
                    mask = torch.sigmoid(logits).unsqueeze(1)
                    conditioned_feat = feat * (1.0 + mask)
                    conditioned_maps.append(conditioned_feat)
                roi_maps_list = conditioned_maps

        # Stack scales → [N, S, C, H, W]
        x = torch.stack(roi_maps_list, dim=1)
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [N, H, W, S, C]
        x = x.view(-1, num_scales, C)              # flatten spatial dims → [N*H*W, S, C]

        # Add positional encoding for scale order
        x = x + self.pos_embed[:, :num_scales, :]

        # ====== Cross-Attention Layer 1 ======
        x1 = self.norm1a(x)
        attn1_out, _ = self.cross_attn1(x1, x1, x1)
        x = x + attn1_out
        x = x + self.ffn1(self.norm1b(x))

        # ====== Cross-Attention Layer 2 ======
        x2 = self.norm2a(x)
        attn2_out, _ = self.cross_attn2(x2, x2, x2)
        x = x + attn2_out
        x = x + self.ffn2(self.norm2b(x))

        # Mean fusion across scales
        fused = x.mean(dim=1)                   # [N*H*W, C]
        fused = fused.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

        fused = self.dropout2d(fused)
        return self.out_proj(fused)




# ---------------------------------------------------------------------
# Full Multi-Scale Saliency MAFormer with ROI-based moding
# ---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------
# Multi-Scale Saliency MAFormer (ROI-mod Version)
# ----------------------------------------------------------
class ClassHead(nn.Module):
    def __init__(self, feature_dim=256, num_classes=90, dropout_p=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)  # [N_rois, num_classes]


class SalientRegionExtractionNetwork(nn.Module):
    """
    Multi-scale fusion network that uses ROI-conditioned mods
    generated by the ModInjectedBackbone backbone.
    """
    def __init__(self, backbone_pretrained=True, mod_injection=True, out_channels=256, pool_size=7, dropout_p=0.2, num_classes=90):
        super().__init__()

        # --- moded Backbone (ResNet + ROI mod injection) ---
        self.backbone = ModInjectedBackbone(backbone_pretrained, mod_injection)

        ch = {'s1': 256, 's2': 512, 's3': 1024, 's4': 2048}

        # --- Channel projections for bottom-up upsampling ---
        self.proj_s1 = nn.Conv2d(ch['s1'], ch['s2'], 1, bias=False)
        self.proj_s2 = nn.Conv2d(ch['s2'], ch['s3'], 1, bias=False)
        self.proj_s3 = nn.Conv2d(ch['s3'], ch['s4'], 1, bias=False)

        # --- Per-scale deformable alignment ---
        self.align_s1 = DeformAlignBlock(ch['s1'])
        self.align_s2 = DeformAlignBlock(ch['s2'])
        self.align_s3 = DeformAlignBlock(ch['s3'])
        self.align_s4_raw = DeformAlignBlock(ch['s4'])
        self.align_s4_fused = DeformAlignBlock(ch['s4'])

        # --- Per-scale ROI heads (multi-scale region encoding) ---
        self.h_s1 = BSDHeadMAF(ch['s1'], out_channels, pool_size, dropout_p)
        self.h_s2 = BSDHeadMAF(ch['s2'], out_channels, pool_size, dropout_p)
        self.h_s3 = BSDHeadMAF(ch['s3'], out_channels, pool_size, dropout_p)
        self.h_s4 = BSDHeadMAF(ch['s4'], out_channels, pool_size, dropout_p)

        # --- Multi-scale attention-based fusion ---
        self.scale_fuser = CrossScaleAttentionFusion(
            out_channels,
            num_scales=4,
            dropout=dropout_p,
            phrase_dim=512
        )

        # --- Tail: global fusion + projection head ---
        self.roi_tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        self.out_head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )

        # --- Saliency mask head ---
        self.mask_head = MaskHead(in_channels=ch['s4'], mid_channels=256, out_channels=1)

        # --- Class head (applied directly to ROI features from backbone) ---
        self.class_head = ClassHead(feature_dim=out_channels, num_classes=num_classes, dropout_p=dropout_p)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, x, rois, phrases):
        """
        x: [B, 3, H, W]
        rois: [N_rois, 5] = (batch_idx, x1, y1, x2, y2)
        returns:
            - roi_embed: fused ROI embeddings [N_rois, out_channels]
            - mask: predicted saliency mask [B, 1, H, W]
            - class_logits: [N_rois, num_classes]
            - phrase_saliency_scores: [N_rois, 1], phrase-conditioned scores used to build ROI overlay.
        """
        if rois is not None:
            rois = rois.to(x.device)
        # todo: send phrases to backbone network
        feats = self.backbone(x, rois, phrases)
        phrase_saliency_scores = feats.get("roi_att_mask", None)
        if phrase_saliency_scores is None:
            phrase_saliency_scores = x.new_zeros((0, 1))
        else:
            phrase_saliency_scores = phrase_saliency_scores.to(x.device)
        maf_phrase_context = feats.get('roi_phrase_context', None)
        raw_phrase_tokens = feats.get('matched_raw_tokens', None)
        if (
            maf_phrase_context is None
            or (not torch.is_tensor(maf_phrase_context))
            or maf_phrase_context.numel() == 0
            or maf_phrase_context.dim() != 2
            or rois is None
            or maf_phrase_context.size(0) != rois.shape[0]
            or maf_phrase_context.size(1) != 512
        ):
            maf_phrase_context = None
        else:
            maf_phrase_context = maf_phrase_context.to(x.device)

        # ---- Top-down saliency mask prediction ----
        f4_raw = self.align_s4_raw(feats['s4'])
        H, W = x.shape[-2:]
        mask = self.mask_head(f4_raw, target_size=(H, W))

        # ---- ROI feature extraction (direct from backbone stage 4) ----
        # Option: could also use a fused feature like f3a or f2a
        # ROI pooling for classification before fusion
        roi_features = self.h_s4.forward_map(
            f4_raw, rois, (H, W), phrase_context=maf_phrase_context
        )  # [N_rois, C, P, P]
        if roi_features is not None and roi_features.numel() > 0:
            pooled_roi_feats = self.roi_tail(roi_features)  # [N_rois, out_channels]
            class_logits = self.class_head(pooled_roi_feats)  # [N_rois, num_classes]
        else:
            pooled_roi_feats = x.new_zeros((0, self.roi_tail[2].out_features))
            class_logits = x.new_zeros((0, self.class_head.classifier[-1].out_features))

        # ---- Multi-scale feature fusion for saliency embeddings ----
        f1a = self.align_s1(feats['s1'])
        f1u = self.proj_s1(f1a)
        f2a = self.align_s2(
            feats['s2']
            + F.interpolate(
                f1u,
                size=feats['s2'].shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        )
        f2u = self.proj_s2(f2a)
        f3a = self.align_s3(feats['s3'] + F.interpolate(f2u, size=feats['s3'].shape[-2:], mode='bilinear', align_corners=False))

        f3u = self.proj_s3(f3a)
        f4_fused_input = feats['s4'] + F.interpolate(
            f3u,
            size=feats['s4'].shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        f4a = self.align_s4_fused(f4_fused_input)

        # ---- Per-scale ROI feature maps for saliency reasoning ----
        m1 = self.h_s1.forward_map(f1a, rois, (H, W), phrase_context=maf_phrase_context)
        m2 = self.h_s2.forward_map(f2a, rois, (H, W), phrase_context=maf_phrase_context)
        m3 = self.h_s3.forward_map(f3a, rois, (H, W), phrase_context=maf_phrase_context)
        m4 = self.h_s4.forward_map(f4a, rois, (H, W), phrase_context=maf_phrase_context)

        roi_maps = [m for m in [m1, m2, m3, m4] if m is not None and m.numel() > 0]
        if len(roi_maps) == 0:
            return (
                x.new_zeros((0, self.out_head[0].out_features)),
                mask,
                class_logits,
                phrase_saliency_scores,
            )

        if raw_phrase_tokens is not None:
            raw_phrase_tokens = raw_phrase_tokens.to(x.device)
            # Ensure token rows match ROI count; otherwise fall back to visual-only fusion.
            if raw_phrase_tokens.numel() == 0 or raw_phrase_tokens.shape[0] != roi_maps[0].shape[0]:
                raw_phrase_tokens = None

        fused_roi_map = self.scale_fuser(
            roi_maps,
            raw_phrase_tokens=raw_phrase_tokens,
        )
        fused_vec = self.roi_tail(fused_roi_map)  # [N_rois, out_channels]
        roi_embed = self.out_head(fused_vec)

        return roi_embed, mask, class_logits, phrase_saliency_scores

    
# ----------------------------------------------------------
# ROI-Aware mod Decoder (with adaptive positional embedding)
# ----------------------------------------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math

class ModDecoder(nn.Module):

    def __init__(
        self,
        in_ch: int = 3,
        embed_dim: int = 512,
        width: int = 256,
        depth: int = 3,
        separable: bool = True,
        resize_to: int = 16,
        ff_dim: int = 512,
        dropout: float = 0.1,
        device="cuda:1",
        num_heads: int = 8,
        num_attn_layers: int = 1,
        max_phrases: int = 50
    ):
        super().__init__()

        self.resize_to = resize_to
        self.device = device
        self.num_heads = num_heads
        self.num_attn_layers = num_attn_layers
        self.max_phrases = max_phrases

        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        text_dim = self.clip_model.text_projection.shape[1]
        self.text_dim = text_dim

        self.stem = ConvBNAct(in_ch, width, k=3, s=1, p=1)
        blocks = []
        for _ in range(depth):
            if separable:
                blocks.append(ConvBNAct(width, width, k=3, s=1, p=1, groups=width))
                blocks.append(ConvBNAct(width, width, k=1, s=1, p=0))
            else:
                blocks.append(ConvBNAct(width, width, k=3, s=1, p=1))
        self.body = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)

        self.roi_visual_encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
        )

        self.fc_visual = nn.Sequential(
            nn.Linear(width, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.position_embed = nn.Embedding(max_phrases, text_dim)

        self.fuse_mlp = nn.Sequential(
            nn.Linear(width + text_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.length_bias_mlp = nn.Sequential(
            nn.Linear(1, ff_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 4, 1)
        )
        self.length_bias_scale = nn.Parameter(torch.tensor(0.0))

        # ── Removed: phrase_to_roi_attn, phrase_query_proj,
        #             roi_key_proj, roi_value_proj
        # ── Saliency uses raw token IDs downstream (see ModInjectedBackbone).

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        roi_crops,
        phrases=None,
        roi_batch_idx=None,
        return_roi_attention=False,
        return_roi_phrase_context=False
    ):
        # ── Visual encoding ──────────────────────────────────────────────
        roi_crops_resized = F.adaptive_avg_pool2d(roi_crops, (self.resize_to, self.resize_to))
        x = roi_crops_resized
        x = self.stem(x)
        x = self.body(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)

        # ── No phrases: return visual-only embeddings ────────────────────
        if phrases is None:
            mod_embeds = self.fc_visual(x)
            N          = roi_crops.size(0)
            seq_len    = self.clip_model.context_length
            dummy_ids  = torch.zeros(N, seq_len, dtype=torch.long, device=roi_crops.device)
            dummy_phrase_embs = torch.zeros(
                N,
                self.text_dim,
                device=roi_crops.device,
                dtype=roi_crops.dtype,
            )
            return mod_embeds, dummy_ids, roi_crops_resized, dummy_phrase_embs

        # CLIP is used only for ROI→phrase matching here. Prompt-score computation uses raw token IDs downstream.

        # ── CLIP image encoding for ROI crops ────────────────────────────
        with torch.no_grad():
            roi_for_clip = F.interpolate(roi_crops, (224, 224), mode="bilinear")
            clip_roi_emb = self.clip_model.encode_image(roi_for_clip)
            clip_roi_emb = clip_roi_emb / clip_roi_emb.norm(dim=-1, keepdim=True)

        # ── CLIP text encoding (global only; raw token IDs stored separately) ─
        text_emb_batches    = []   # [P, 512]       global pooled, per batch item
        raw_tokens_batches  = []   # [P, seq_len]   raw token ids (for downstream saliency)
        length_batches      = []

        max_clip_text_len = max(1, int(self.clip_model.context_length) - 2)
        denom_log  = math.log1p(max_clip_text_len)
        clip_device = next(self.clip_model.parameters()).device

        for plist in phrases:
            if len(plist) == 0:
                text_emb_batches.append(None)
                raw_tokens_batches.append(None)
                length_batches.append(None)
                continue

            tokens = clip.tokenize(plist, truncate=True).to(clip_device)

            with torch.no_grad():
                t = self.clip_model.encode_text(tokens)   # [P, 512] global (matching only)

            t = t / t.norm(dim=-1, keepdim=True)

            text_emb_batches.append(t)
            raw_tokens_batches.append(tokens)

            non_pad   = (tokens != 0).sum(dim=1).float()
            token_len = (non_pad - 2.0).clamp(min=1.0)
            len_norm  = torch.log1p(token_len) / denom_log
            length_batches.append(len_norm.unsqueeze(-1))

        # ── Normalise roi_batch_idx ──────────────────────────────────────
        if roi_batch_idx is None:
            roi_batch_idx = torch.zeros(
                clip_roi_emb.size(0), dtype=torch.long, device=roi_crops.device
            )
        else:
            roi_batch_idx = roi_batch_idx.to(device=roi_crops.device, dtype=torch.long)

        max_batch_idx = int(roi_batch_idx.max().item()) if roi_batch_idx.numel() > 0 else -1
        while len(text_emb_batches) <= max_batch_idx:
            text_emb_batches.append(None)
            raw_tokens_batches.append(None)
            length_batches.append(None)

        # ── Per-ROI phrase matching (cosine sim in CLIP space) ───────────
        aligned_phrase_embs    = []
        phrase_indices         = []
        matched_raw_tokens     = []   # [N, seq_len]
        matched_roi_crops      = []   # [N, 3, resize_to, resize_to]

        seq_len = self.clip_model.context_length

        for i in range(clip_roi_emb.size(0)):
            b = roi_batch_idx[i].item()

            if text_emb_batches[b] is None:
                # No phrases for this image — fill with zeros
                aligned_phrase_embs.append(torch.zeros_like(clip_roi_emb[i]))
                phrase_indices.append(0)
                matched_raw_tokens.append(
                    torch.zeros(seq_len, dtype=torch.long, device=clip_device)
                )
                matched_roi_crops.append(roi_crops_resized[i])
                continue

            t = text_emb_batches[b]    # [P, 512]
            l = length_batches[b]      # [P, 1]
            r = clip_roi_emb[i].unsqueeze(0)   # [1, 512]

            # Cosine similarity + learned length bias → soft match weights
            sim          = (r @ t.T) / math.sqrt(clip_roi_emb.size(-1))
            length_bias  = self.length_bias_mlp(l.float()).T.to(dtype=sim.dtype)
            sim          = sim + self.length_bias_scale.to(dtype=sim.dtype) * length_bias
            weight       = sim.softmax(dim=-1)   # [1, P]

            best_idx = weight.argmax(dim=-1).item()
            phrase_indices.append(best_idx)

            # Weighted-average phrase embedding for mod_embeds
            t_star = weight @ t
            aligned_phrase_embs.append(t_star.squeeze(0))

            matched_raw_tokens.append(raw_tokens_batches[b][best_idx])  # [seq_len]
            matched_roi_crops.append(roi_crops_resized[i])

        aligned_phrase_embs = torch.stack(aligned_phrase_embs, dim=0)   # [N, 512]
        matched_raw_tokens  = torch.stack(matched_raw_tokens,  dim=0)   # [N, seq_len]
        matched_roi_crops   = torch.stack(matched_roi_crops,   dim=0)   # [N, 3, resize_to, resize_to]

        # ── Positional embedding (phrase slot index) ─────────────────────
        phrase_indices_tensor = torch.tensor(
            phrase_indices, device=aligned_phrase_embs.device, dtype=torch.long
        )
        aligned_phrase_embs = aligned_phrase_embs + self.position_embed(phrase_indices_tensor)

        # ── Multimodal fusion for mod_embeds ─────────────────────────────
        roi_feat      = clip_roi_emb.to(roi_crops.dtype)
        text_feat     = aligned_phrase_embs.to(roi_crops.dtype)
        joint_text_roi = roi_feat * text_feat

        multimodal = torch.cat([x, joint_text_roi], dim=-1)
        mod_embeds = self.fuse_mlp(multimodal)   # [N, embed_dim]

        return mod_embeds, matched_raw_tokens, matched_roi_crops, aligned_phrase_embs


# ----------------------------------------------------------
# ResNet backbone with colour-overlay visual prompting
# ----------------------------------------------------------
class ModInjectedBackbone(nn.Module):
    """
    ResNet-101 encoder with input-space colour-overlay visual prompting.

    Saliency scores: ROI query token cross-attends to trainable embeddings of
    raw phrase token IDs (CLIP is only used in ModDecoder for ROI→phrase matching).
    """

    def __init__(self, backbone_pretrained=True, mod_injection=True):
        super().__init__()
        self.device_main = torch.device("cuda:1")
        self.device_clip = torch.device("cuda:1")
        _ = mod_injection
        self.visual_prompting = True

        # ── Frozen ResNet-101 backbone ───────────────────────────────────
        self.backbone = resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2 if backbone_pretrained else None
        ).to(self.device_main)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # ── ModDecoder (on device_clip) ──────────────────────────────────
        self.mod_decoder = ModDecoder(
            embed_dim=512,
            num_heads=4,
            num_attn_layers=3,
            device=str(self.device_clip),
        ).to(self.device_clip)

        self.prompt_max_strength = 0.5

        # ── Colourmap: deep-blue → cyan → yellow → red ───────────────────
        _cmap = torch.tensor([
            [0.10, 0.05, 0.60],   # 0.00 – deep blue  (low saliency)
            [0.00, 0.70, 0.95],   # 0.33 – cyan
            [0.95, 0.90, 0.00],   # 0.67 – yellow
            [0.85, 0.05, 0.05],   # 1.00 – red        (high saliency)
        ], dtype=torch.float32)
        self.register_buffer("_cmap_stops", _cmap)

        self.roi_feature_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
        ).to(self.device_clip)
        self.raw_token_embed = nn.Embedding(49408, 512, padding_idx=0).to(self.device_clip)
        self.raw_token_pos_embed = nn.Embedding(77, 512).to(self.device_clip)
        self.raw_phrase_token_norm = nn.LayerNorm(512).to(self.device_clip)
        self.raw_phrase_ffn = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        ).to(self.device_clip)
        self.saliency_cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        ).to(self.device_clip)
        self.saliency_head = nn.Sequential(
            nn.LayerNorm(512 * 4),
            nn.Linear(512 * 4, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        ).to(self.device_clip)

        # Freeze ROI feature encoder and keep it in eval mode so BatchNorm
        # layers work even for single-ROI batches during saliency scoring.
        # for p in self.roi_feature_encoder.parameters():
        #     p.requires_grad = False
        # self.roi_feature_encoder.eval()

    def train(self, mode: bool = True):
        """
        Override .train() so that roi_feature_encoder stays in eval mode.
        This avoids BatchNorm errors when the ROI batch size is 1, while
        allowing the rest of the backbone to toggle train/eval normally.
        """
        super().train(mode)
        self.roi_feature_encoder.eval()
        return self

    # ------------------------------------------------------------------
    # ROI ↔ phrase saliency (trainable raw token embeds; CLIP not used here)
    # ------------------------------------------------------------------
    def _compute_saliency_scores(
        self,
        raw_tokens: torch.Tensor,
        roi_crops: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.raw_token_embed.parameters()).device

        raw_tokens = raw_tokens.to(device=device, dtype=torch.long)
        roi_crops = roi_crops.to(device=device)

        N, L = raw_tokens.shape
        head_dtype = next(self.saliency_head.parameters()).dtype

        scores_out = torch.zeros(N, 1, device=device, dtype=head_dtype)

        valid_rows = (raw_tokens != 0).any(dim=1)

        if not valid_rows.any():
            return scores_out

        raw_tokens_valid = raw_tokens[valid_rows]
        roi_crops_valid = roi_crops[valid_rows]

        pad_mask = raw_tokens_valid == 0

        token_emb = self.raw_token_embed(raw_tokens_valid)

        pos_ids = torch.arange(
            L,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(raw_tokens_valid.size(0), L)

        token_emb = token_emb + self.raw_token_pos_embed(pos_ids)

        token_emb = self.raw_phrase_token_norm(token_emb)
        token_emb = self.raw_phrase_ffn(token_emb)

        token_emb = torch.nan_to_num(
            token_emb,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        enc_dtype = next(self.roi_feature_encoder.parameters()).dtype
        roi_vec = self.roi_feature_encoder(roi_crops_valid.to(dtype=enc_dtype))

        roi_vec = torch.nan_to_num(
            roi_vec,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        attn_dtype = next(self.saliency_cross_attn.parameters()).dtype

        roi_query = roi_vec.unsqueeze(1).to(dtype=attn_dtype)
        phrase_tokens = token_emb.to(dtype=attn_dtype)

        cross_out, _ = self.saliency_cross_attn(
            query=roi_query,
            key=phrase_tokens,
            value=phrase_tokens,
            key_padding_mask=pad_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        cross_out = torch.nan_to_num(
            cross_out,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        roi_vec = roi_query.squeeze(1)
        phrase_vec = cross_out.squeeze(1)

        fused = torch.cat(
            [
                roi_vec,
                phrase_vec,
                roi_vec * phrase_vec,
                torch.abs(roi_vec - phrase_vec),
            ],
            dim=-1,
        )

        fused = torch.nan_to_num(
            fused,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        scores_valid = self.saliency_head(fused.to(dtype=head_dtype))

        scores_valid = torch.nan_to_num(
            scores_valid,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        scores_out[valid_rows] = scores_valid

        return scores_out

    # ------------------------------------------------------------------
    # Differentiable colourmap
    # ------------------------------------------------------------------
    def _apply_colormap(self, t: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear interpolation over _cmap_stops. [B,1,H,W] → [B,3,H,W]."""
        stops  = self._cmap_stops.to(t)
        K      = stops.shape[0]
        t_flat = t.squeeze(1)                          # [B, H, W]
        t_s    = t_flat * (K - 1)
        lo     = t_s.long().clamp(0, K - 2)
        hi     = (lo + 1).clamp(max=K - 1)
        frac   = (t_s - lo.float()).unsqueeze(-1)      # [B, H, W, 1]
        rgb    = stops[lo] + frac * (stops[hi] - stops[lo])
        return rgb.permute(0, 3, 1, 2)                 # [B, 3, H, W]

    # ------------------------------------------------------------------
    # Colour overlay
    # ------------------------------------------------------------------
    def _apply_visual_prompt(self, x: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        """
        Blend a colour heatmap onto x using per-pixel saliency as alpha.
            low  saliency → deep blue
            high saliency → red
        """
        B    = x.shape[0]
        flat = prompt_mask.view(B, -1)
        # mn   = flat.min(dim=1).values.view(B, 1, 1, 1)
        # mx   = flat.max(dim=1).values.view(B, 1, 1, 1)

        # Per-image contrast stretch → always spans full [0, 1] colour range
        # norm_mask      = (prompt_mask - mn) / (mx - mn + 1e-6)
        norm_mask = prompt_mask.clamp(0.0, 1.0)
        colour_overlay = self._apply_colormap(norm_mask)           # [B, 3, H, W]

        # Raw sigmoid score as alpha: background (≈0) untouched
        alpha           = prompt_mask
        return x + self.prompt_max_strength * alpha * colour_overlay

    # ------------------------------------------------------------------
    # Prompt mask builder
    # ------------------------------------------------------------------
    def _build_visual_prompt_mask(self, x, rois, phrases=None, output_size=64):
        B, _, H, W = x.shape
        device = x.device

        _zero_mask    = x.new_zeros(B, 1, H, W)
        _zero_context = x.new_zeros(0, self.mod_decoder.text_dim)
        _zero_scores  = x.new_zeros(0, 1)
        seq_len = int(self.mod_decoder.clip_model.context_length)
        _zero_tokens  = torch.zeros(0, seq_len, dtype=torch.long, device=device)

        if rois is None:
            return _zero_mask, _zero_context, _zero_scores, _zero_tokens

        if isinstance(rois, (list, tuple)):
            rois = torch.as_tensor(rois, dtype=torch.float32, device=device)
        else:
            rois = rois.to(device=device, dtype=torch.float32)

        if rois.numel() == 0:
            return _zero_mask, _zero_context, _zero_scores, _zero_tokens

        batch_idx = rois[:, 0].long().clamp(0, B - 1)
        x1 = rois[:, 1].clamp(0.0, float(W))
        y1 = rois[:, 2].clamp(0.0, float(H))
        x2 = rois[:, 3].clamp(0.0, float(W))
        y2 = rois[:, 4].clamp(0.0, float(H))

        valid = (x2 > x1) & (y2 > y1)
        if not valid.any():
            return _zero_mask, _zero_context, _zero_scores, _zero_tokens

        rois_clamped   = torch.stack(
            [batch_idx.to(dtype=torch.float32), x1, y1, x2, y2], dim=1
        )
        rois_for_align = rois_clamped[valid]
        idx            = rois_for_align[:, 0].long().clamp(0, B - 1)
        rois_for_align = torch.cat(
            [idx.unsqueeze(1).to(rois_for_align.dtype), rois_for_align[:, 1:]], dim=1
        )

        roi_crops = roi_align(
            x, rois_for_align,
            output_size=(output_size, output_size),
            aligned=True,
        )

        decoder_device = next(self.mod_decoder.parameters()).device
        roi_crops = roi_crops.to(decoder_device)
        idx       = idx.to(decoder_device)

        # ── ModDecoder: mod_embeds + matched token ids (for visual-text alignment) ──
        mod_embeds, matched_raw_tokens, matched_roi_crops, matched_phrase_embs = self.mod_decoder(
            roi_crops, phrases=phrases,
            roi_batch_idx=idx, return_roi_attention=True,
        )
        # matched_raw_tokens : [N, seq_len] on device_clip — CLIP-best phrase per ROI

        # ── Saliency from per-ROI matched phrase token IDs + ROI crops ─────
        if matched_raw_tokens.numel() > 0:
            saliency_scores = self._compute_saliency_scores(
                matched_raw_tokens, matched_roi_crops
            )   # [N, 1]  on device_clip
        else:
            saliency_scores = x.new_zeros(0, 1).to(decoder_device)

        mod_embeds      = mod_embeds.to(x.device)
        matched_phrase_embs = matched_phrase_embs.to(x.device)
        saliency_scores = saliency_scores.to(x.device)
        roi_phrase_context = matched_phrase_embs
        matched_raw_tokens = matched_raw_tokens.to(x.device)

        # ── Splat per-ROI saliency scores onto the image canvas ───────────
        prompt_mask = x.new_zeros(B, 1, H, W)

        for i in range(rois_for_align.size(0)):
            b   = int(idx[i].item())
            w   = saliency_scores[i, 0].to(prompt_mask.device)
            rx1, ry1, rx2, ry2 = (rois_for_align[i, j] for j in (1, 2, 3, 4))
            x1i = int(rx1.floor().clamp(0, W - 1).item())
            y1i = int(ry1.floor().clamp(0, H - 1).item())
            x2i = min(int(rx2.ceil().clamp(0.0, float(W)).item()), W)
            y2i = min(int(ry2.ceil().clamp(0.0, float(H)).item()), H)
            if x2i <= x1i or y2i <= y1i:
                continue
            prompt_mask[b, :, y1i:y2i, x1i:x2i] += w

        # prompt_mask = torch.sigmoid(prompt_mask)
        prompt_mask = prompt_mask.clamp(0.0, 1.0)
        return prompt_mask, roi_phrase_context, saliency_scores, matched_raw_tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, rois=None, phrases=None):
        prompt_mask, roi_phrase_context, roi_att_mask, matched_raw_tokens = \
            self._build_visual_prompt_mask(x, rois, phrases)
        x_in = self._apply_visual_prompt(x, prompt_mask)

        x = self.backbone.conv1(x_in)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        s1 = self.backbone.layer1(x)
        s2 = self.backbone.layer2(s1)
        s3 = self.backbone.layer3(s2)
        s4 = self.backbone.layer4(s3)

        return {
            's1': s1, 's2': s2, 's3': s3, 's4': s4,
            'prompt_mask':        prompt_mask,
            'roi_phrase_context': roi_phrase_context,
            'roi_att_mask':       roi_att_mask,
            'matched_raw_tokens': matched_raw_tokens,
        }


class MaskHead(nn.Module):
    """
    Full-resolution mask prediction head.
    Upsamples stride-32 s4 feature map to input HxW.
    """
    def __init__(self, in_channels=2048, mid_channels=256, out_channels=1):
        super().__init__()

        # 1/32 → 1/16
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, 2, 2),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        # 1/16 → 1/8
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        # 1/8 → 1/4
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        # 1/4 → 1/2
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        # 1/2 → 1/1 (full res)
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2),  # final double-upsample
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 1)
        )

    def forward(self, x, target_size):
        """
        x: s4 feature map [B, C, H/32, W/32]
        target_size: (H, W) - full resolution of the input image
        """
        x = self.up1(x)   # 1/32 → 1/16
        x = self.up2(x)   # 1/16 → 1/8
        x = self.up3(x)   # 1/8 → 1/4
        x = self.up4(x)   # 1/4 → 1/2
        x = self.final_conv(x)  # 1/2 → 1/1

        # Ensure exact spatial match
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(x)
