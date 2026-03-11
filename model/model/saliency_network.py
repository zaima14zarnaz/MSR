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
from torch_geometric.nn import GATConv
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
        self.maf = MAFBlock(out_channels, dropout=dropout_p)
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

    def _roi_align_and_project(self, feat_map, rois, image_shape):
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
        x = self.maf(x)         # [N_rois, C, P, P]
        return x

    # NEW: expose pre-GAP map
    def forward_map(self, feat_map, rois, image_shape):
        return self._roi_align_and_project(feat_map, rois, image_shape)

    # # Original behavior (kept)
    # def forward(self, feat_map, rois, image_shape):
    #     x = self._roi_align_and_project(feat_map, rois, image_shape)
    #     if x.size(0) == 0:
    #         return feat_map.new_zeros((0, self.out_channels))
    #     return self.head(x)
    
# ----------------------------------------------
# MAFormer-style local/global fusion (per ROI)
# ----------------------------------------------
class MAFBlock(nn.Module):
    """
    Simplified MAFBlock (No Global Path 2 or 3)
    -------------------------------------------
    - Builds 4 path features (3 local + 1 global)
    - Each spatial location has a 4-token sequence (one per path)
    - Transformer Encoder (4 layers) aggregates cross-path relationships
    - GATConv fuses token-level graph
    - Optional downsampling (token_stride) for efficiency
    """

    def __init__(self, channels: int, dropout: float = 0.0,
                 num_heads: int = 4, token_stride: int = 1):
        super().__init__()
        self.channels = channels
        self.token_stride = token_stride

        # ===== Local feature branches =====
        self.local_k1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
        )
        self.local_k2 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
        )
        self.local_k3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
        )

        # ===== Single global feature branch =====
        self.global_k1 = nn.Sequential(
            nn.Conv2d(channels, channels, 15, padding=7, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
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

        # ===== GAT-based Fusion =====
        self.gat_fuser = GATConv(channels, channels, heads=num_heads, concat=False, dropout=dropout)

        # ===== Output projection =====
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    @staticmethod
    def _make_complete_edge_index(num_nodes, device):
        idx = torch.arange(num_nodes, device=device)
        src = idx.repeat_interleave(num_nodes)
        dst = idx.repeat(num_nodes)
        return torch.stack([src, dst], dim=0)

    def forward(self, x):
        N, C, H, W = x.shape

        # ----- Local paths -----
        x_l1 = self.local_k1(x)
        x_l2 = self.local_k2(x)
        x_l3 = self.local_k3(x)

        # ----- Single global path -----
        x_g1 = self.global_k1(x)

        # stack 4 paths: [N, 4, C, H, W]
        all_feats = torch.stack([x_l1, x_l2, x_l3, x_g1], dim=1)

        # optional downsampling
        if self.token_stride > 1:
            all_feats = all_feats.view(N * 4, C, H, W)
            all_feats = self.shrink(all_feats)
            Hs, Ws = all_feats.shape[-2:]
            all_feats = all_feats.view(N, 4, C, Hs, Ws)
        else:
            Hs, Ws = H, W

        # reshape for transformer: each (H,W) location has 4 path tokens
        feats = all_feats.permute(0, 3, 4, 1, 2).contiguous()  # [N, Hs, Ws, 4, C]
        G = N * Hs * Ws
        tokens = feats.view(G, 4, C)  # sequence length = 4

        # Transformer fusion across 4 paths
        fused_tokens = self.transformer_fusion(tokens)  # [G, 4, C]

        # GAT-based fusion across 4 tokens per location
        base_edge = self._make_complete_edge_index(4, x.device)
        offsets = torch.arange(G, device=x.device).repeat_interleave(base_edge.shape[1]) * 4
        edge_index = base_edge.repeat(1, G) + offsets

        node_feats = fused_tokens.view(G * 4, C)
        out = self.gat_fuser(node_feats, edge_index)
        out = out.view(G, 4, C).mean(dim=1)  # mean over fused paths

        # reshape back to image-like format
        fused = out.view(N, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
        if (Hs, Ws) != (H, W):
            fused = F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=False)

        return self.out_proj(fused) + x


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
    def __init__(self, channels: int, num_scales: int = 4, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.num_heads = num_heads

        # 1️⃣ Projection to reduce computational cost before attention
        self.scale_proj = nn.Linear(channels, channels)
        # self.pos_embed = nn.Parameter(torch.randn(1, num_scales, channels))

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

        # Learnable gating between layers (stabilizes multi-layer attention)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.dropout2d = nn.Dropout2d(dropout)

    def forward(self, roi_maps_list):
        if len(roi_maps_list) == 0:
            raise ValueError("roi_maps_list is empty")

        N, C, H, W = roi_maps_list[0].shape
        num_scales = len(roi_maps_list)

        # Stack scales → [N, S, C, H, W]
        x = torch.stack(roi_maps_list, dim=1)
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [N, H, W, S, C]
        x = x.view(-1, num_scales, C)              # flatten spatial dims → [N*H*W, S, C]

        # Add positional encoding for scale order
        # x = x + self.pos_embed[:, :num_scales, :]

        # ====== Cross-Attention Layer 1 ======
        x1 = self.norm1a(x)
        attn1_out, _ = self.cross_attn1(x1, x1, x1)
        x = x + attn1_out
        x = x + self.ffn1(self.norm1b(x))

        # ====== Cross-Attention Layer 2 ======
        x2 = self.norm2a(x)
        attn2_out, _ = self.cross_attn2(x2, x2, x2)
        x = x + self.alpha * attn2_out          # gated residual
        x = x + self.ffn2(self.norm2b(x))

        # Mean fusion across scales
        fused = x.mean(dim=1)                   # [N*H*W, C]
        fused = fused.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

        fused = self.dropout2d(fused)
        return self.out_proj(fused)




# ---------------------------------------------------------------------
# Full Multi-Scale Saliency MAFormer with ROI-based filming
# ---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152

# ----------------------------------------------------------
# Multi-Scale Saliency MAFormer (ROI-film Version)
# ----------------------------------------------------------
class ClassHead(nn.Module):
    def __init__(self, feature_dim=256, num_classes=91, dropout_p=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x):
        # x: [N_rois, feature_dim]
        return self.classifier(x)


class SalientRegionExtractionNetwork(nn.Module):
    """
    Multi-scale fusion network that uses ROI-conditioned films
    generated by the FiLMInjectedBackbone backbone.
    """
    def __init__(self, backbone_pretrained=True, film_injection=True, out_channels=256, pool_size=7, dropout_p=0.2, num_classes=90):
        super().__init__()

        # --- filmed Backbone (ResNet + ROI film injection) ---
        self.backbone = FiLMInjectedBackbone(backbone_pretrained, film_injection)

        ch = {'s1': 256, 's2': 512, 's3': 1024, 's4': 2048}

        # --- Channel projections for bottom-up upsampling ---
        self.proj_s1 = nn.Conv2d(ch['s1'], ch['s2'], 1, bias=False)
        self.proj_s2 = nn.Conv2d(ch['s2'], ch['s3'], 1, bias=False)
        self.proj_s3 = nn.Conv2d(ch['s3'], ch['s4'], 1, bias=False)

        # --- Per-scale deformable alignment ---
        self.align_s1 = DeformAlignBlock(ch['s1'])
        self.align_s2 = DeformAlignBlock(ch['s2'])
        self.align_s3 = DeformAlignBlock(ch['s3'])
        self.align_s4 = DeformAlignBlock(ch['s4'])

        # --- Per-scale ROI heads (multi-scale region encoding) ---
        self.h_s1 = BSDHeadMAF(ch['s1'], out_channels, pool_size, dropout_p)
        self.h_s2 = BSDHeadMAF(ch['s2'], out_channels, pool_size, dropout_p)
        self.h_s3 = BSDHeadMAF(ch['s3'], out_channels, pool_size, dropout_p)
        self.h_s4 = BSDHeadMAF(ch['s4'], out_channels, pool_size, dropout_p)

        # --- Multi-scale attention-based fusion ---
        self.scale_fuser = CrossScaleAttentionFusion(out_channels, num_scales=4, dropout=dropout_p)

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
    def forward(self, x, rois, phrases=None):
        """
        x: [B, 3, H, W]
        rois: [N_rois, 5] = (batch_idx, x1, y1, x2, y2)
        returns:
            - roi_embed: fused ROI embeddings [N_rois, out_channels]
            - mask: predicted saliency mask [B, 1, H, W]
            - class_logits: [N_rois, num_classes]
        """
        # todo: send phrases to backbone network
        feats = self.backbone(x, rois, phrases)
        H, W = x.shape[-2:]

        # ---- Top-down saliency mask prediction ----
        f4 = self.align_s4(feats['s4'])
        H, W = x.shape[-2:]
        mask = self.mask_head(f4, target_size=(H, W))

        # ---- ROI feature extraction (direct from backbone stage 4) ----
        # Option: could also use a fused feature like f3a or f2a
        # ROI pooling for classification before fusion
        roi_features = self.h_s4.forward_map(f4, rois, (H, W))  # [N_rois, C, P, P]
        if roi_features is not None and roi_features.numel() > 0:
            pooled_roi_feats = self.roi_tail(roi_features)  # [N_rois, out_channels]
            class_logits = self.class_head(pooled_roi_feats)  # [N_rois, num_classes]
        else:
            pooled_roi_feats = x.new_zeros((0, self.roi_tail[2].out_features))
            class_logits = x.new_zeros((0, self.class_head.classifier[-1].out_features))

        # ---- Multi-scale feature fusion for saliency embeddings ----
        f1a = self.align_s1(feats['s1'])
        f1u = self.proj_s1(f1a)
        f2a = self.align_s2(feats['s2'] + F.interpolate(f1u, size=feats['s2'].shape[-2:], mode='bilinear', align_corners=False))

        f2u = self.proj_s2(f2a)
        f3a = self.align_s3(feats['s3'] + F.interpolate(f2u, size=feats['s3'].shape[-2:], mode='bilinear', align_corners=False))

        f3u = self.proj_s3(f3a)
        f4a = self.align_s4(feats['s4'] + F.interpolate(f3u, size=feats['s4'].shape[-2:], mode='bilinear', align_corners=False))

        # ---- Per-scale ROI feature maps for saliency reasoning ----
        m1 = self.h_s1.forward_map(f1a, rois, (H, W))
        m2 = self.h_s2.forward_map(f2a, rois, (H, W))
        m3 = self.h_s3.forward_map(f3a, rois, (H, W))
        m4 = self.h_s4.forward_map(f4a, rois, (H, W))

        roi_maps = [m for m in [m1, m2, m3, m4] if m is not None and m.numel() > 0]
        if len(roi_maps) == 0:
            return x.new_zeros((0, self.out_head[0].out_features)), mask, class_logits

        fused_roi_map = self.scale_fuser(roi_maps)
        fused_vec = self.roi_tail(fused_roi_map)  # [N_rois, out_channels]
        roi_embed = self.out_head(fused_vec)

        return roi_embed, mask, class_logits

    
# ----------------------------------------------------------
# ROI-Aware film Decoder (with adaptive positional embedding)
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math

class FiLMDecoder(nn.Module):

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
        device="cuda",
        num_heads: int = 8,
        num_attn_layers: int = 1,
        max_phrases: int = 50    # ⭐ NEW: Max expected phrase count
    ):
        super().__init__()

        self.resize_to = resize_to
        self.device = device
        self.num_heads = num_heads
        self.num_attn_layers = num_attn_layers
        self.max_phrases = max_phrases

        # -------------------------------
        # Load CLIP (Frozen)
        # -------------------------------
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        text_dim = self.clip_model.text_projection.shape[1]

        # -------------------------------
        # CNN ROI visual encoder
        # -------------------------------
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

        self.fc_visual = nn.Sequential(
            nn.Linear(width, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # -------------------------------
        # ⭐ NEW: POSITIONAL EMBEDDINGS
        # -------------------------------
        self.position_embed = nn.Embedding(max_phrases, text_dim)

        # -------------------------------
        # Multimodal fusion MLP
        # -------------------------------
        self.fuse_mlp = nn.Sequential(
            nn.Linear(width + text_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # -------------------------------
        # Cross-attention stack
        # -------------------------------
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=text_dim,
                                  num_heads=num_heads,
                                  batch_first=True)
            for _ in range(num_attn_layers)
        ])


    # ---------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------
    def forward(self, roi_crops, phrases=None, roi_batch_idx=None):

        # -----------------------------
        # CNN visual branch
        # -----------------------------
        x = F.adaptive_avg_pool2d(roi_crops, (self.resize_to, self.resize_to))
        x = self.stem(x)
        x = self.body(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)

        if phrases is None:
            return self.fc_visual(x)

        # -----------------------------
        # CLIP ROI embeddings
        # -----------------------------
        with torch.no_grad():
            roi_for_clip = F.interpolate(roi_crops, (224, 224), mode="bilinear")
            clip_roi_emb = self.clip_model.encode_image(roi_for_clip)
            clip_roi_emb = clip_roi_emb / clip_roi_emb.norm(dim=-1, keepdim=True)

        # -----------------------------
        # CLIP text embeddings
        # -----------------------------
        text_emb_batches = []
        for plist in phrases:
            if len(plist) == 0:
                text_emb_batches.append(None)
                continue

            tokens = clip.tokenize(plist, truncate=True).to(self.device)
            with torch.no_grad():
                t = self.clip_model.encode_text(tokens)
            t = t / t.norm(dim=-1, keepdim=True)
            text_emb_batches.append(t)

        # -----------------------------
        # Dot-product alignment (ROI ↔ phrases)
        # -----------------------------
        aligned_phrase_embs = []
        phrase_indices = []

        for i in range(clip_roi_emb.size(0)):
            b = roi_batch_idx[i].item()

            if text_emb_batches[b] is None:
                aligned_phrase_embs.append(torch.zeros_like(clip_roi_emb[i]))
                phrase_indices.append(0)
                continue

            t = text_emb_batches[b]
            r = clip_roi_emb[i].unsqueeze(0)

            sim = (r @ t.T) / math.sqrt(clip_roi_emb.size(-1))
            weight = sim.softmax(dim=-1)

            # ⭐ NEW: Store the best phrase index per ROI
            best_idx = weight.argmax(dim=-1).item()
            phrase_indices.append(best_idx)

            t_star = weight @ t
            aligned_phrase_embs.append(t_star.squeeze(0))

        aligned_phrase_embs = torch.stack(aligned_phrase_embs, 0)   # [N, 512]

        # -----------------------------
        # ⭐ NEW: ADD POSITIONAL EMBEDDING TO TEXT EMBEDDINGS
        # -----------------------------
        phrase_indices_tensor = torch.tensor(phrase_indices, device=self.device)
        pos_emb = self.position_embed(phrase_indices_tensor)  # [N, 512]

        aligned_phrase_embs = aligned_phrase_embs + pos_emb

        # -----------------------------
        # Cross-attention fusion
        # -----------------------------
        q = clip_roi_emb.unsqueeze(1).to(roi_crops.dtype)
        kv = aligned_phrase_embs.unsqueeze(1).to(roi_crops.dtype)

        attn_out = q
        for attn_layer in self.attn_layers:
            attn_out, _ = attn_layer(attn_out, kv, kv)

        attn_out = attn_out.squeeze(1)

        # -----------------------------
        # Multimodal fusion
        # -----------------------------
        multimodal = torch.cat([x, attn_out], dim=-1)
        return self.fuse_mlp(multimodal)

# ----------------------------------------------------------
# ResNet Backbone with ROI-Conditioned films
# ----------------------------------------------------------
class FiLMInjectedBackbone(nn.Module):
    def __init__(self, backbone_pretrained=True, film_injection=True):
        super().__init__()
        self.film_injection = film_injection

        # Load ResNet-101 backbone
        self.backbone = resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2 if backbone_pretrained else None
        )

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Transformer-based film generator
        self.film_decoder = FiLMDecoder(
            embed_dim=512,
            num_heads=4,
            num_attn_layers=3,
            device="cuda"
        )

        # Layer-specific learnable film embeddings
        self.layer_films = nn.ParameterDict({
            'layer1': nn.Parameter(torch.randn(512)),
            'layer2': nn.Parameter(torch.randn(512)),
            'layer3': nn.Parameter(torch.randn(512)),
            'layer4': nn.Parameter(torch.randn(512)),
        })

        # Project fused+layer films into per-layer channel maps
        self.proj_layers = nn.ModuleDict({
            'layer1': nn.Linear(512, 64),
            'layer2': nn.Linear(512, 256),
            'layer3': nn.Linear(512, 512),
            'layer4': nn.Linear(512, 1024),
        })

    # ----------------- ROI film Fusion -----------------
    def _fuse_roi_films(self, x, rois, phrases=None, output_size=64):
        """
        Returns: fused ROI film [B, 512]
        """
        if isinstance(rois, (list, tuple)):
            rois = torch.as_tensor(rois, dtype=torch.float32, device=x.device)
        else:
            rois = rois.to(device=x.device, dtype=torch.float32)

        B = x.size(0)

        if rois.numel() == 0:
            return x.new_zeros(B, 512)

        idx = rois[:, 0].long().clamp_(0, B - 1)
        rois = torch.cat([idx.unsqueeze(1).to(rois.dtype), rois[:, 1:]], dim=1)

        roi_crops = roi_align(x, rois, output_size=(output_size, output_size), aligned=True)
        film_embeds = self.film_decoder(
            roi_crops,
            phrases=phrases,
            roi_batch_idx=idx
        )  # [N_rois, 512]

        fused = x.new_zeros(B, 512)
        counts = x.new_zeros(B, 1)

        fused.index_add_(0, idx, film_embeds)
        counts.index_add_(0, idx, torch.ones(film_embeds.size(0), 1, device=x.device, dtype=x.dtype))

        fused = fused / counts.clamp_min(1)
        return fused  # [B, 512]

    # ----------------- Injection -----------------
    def _inject(self, feat, fused_film, layer_name):
        """
        feat: [B, C, H, W]
        fused_film: [B, 512]
        layer_name: 'layer1' | 'layer2' | 'layer3' | 'layer4'
        """
        lp = self.layer_films[layer_name]              # [512]
        film = fused_film + lp.unsqueeze(0)            # [B, 512]
        proj = self.proj_layers[layer_name]            # 512 -> C
        p = proj(film).unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
        return feat + p.expand(-1, -1, feat.size(-2), feat.size(-1))

    # ----------------- Forward -----------------
    def forward(self, x, rois=None, phrases=None):
        """
        x:    [B, 3, H, W]
        rois: [N_rois, 5] = (batch_idx, x1, y1, x2, y2)
        """
        # If FiLM injection is disabled: return backbone feature maps directly
        if not self.film_injection:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            s1 = self.backbone.layer1(x)
            s2 = self.backbone.layer2(s1)
            s3 = self.backbone.layer3(s2)
            s4 = self.backbone.layer4(s3)

            return {'s1': s1, 's2': s2, 's3': s3, 's4': s4}

        # Otherwise do FiLM modulation as before
        fused_film = self._fuse_roi_films(x, rois, phrases)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # [B, 64, H1, W1]

        x1 = self._inject(x, fused_film, 'layer1')
        x1 = self.backbone.layer1(x1)

        x2 = self._inject(x1, fused_film, 'layer2')
        x2 = self.backbone.layer2(x2)

        x3 = self._inject(x2, fused_film, 'layer3')
        x3 = self.backbone.layer3(x3)

        x4 = self._inject(x3, fused_film, 'layer4')
        x4 = self.backbone.layer4(x4)

        return {'s1': x1, 's2': x2, 's3': x3, 's4': x4}



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
        # x: [N_rois, feature_dim]
        return self.classifier(x)  # [N_rois, num_classes]


        
