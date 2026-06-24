"""
Model description: Same as variant E but with a class head.

MSRModel supports two call paths:
- Integrated: forward(images, rois, phrases) when salient_region_extractor is set.
- Component analysis: forward(obj_feats, rois=..., phrases=None) on precomputed
  ROI features [N, D] from a separate BSD network (no extractor required).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TransformerRankHead(nn.Module):
    def __init__(self, feature_dim, num_heads=2, depth=4, dropout_p=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dropout=dropout_p,
            activation="gelu",
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.drop = nn.Dropout(dropout_p)
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, feats):
        # feats: [N, D]
        feats = self.tr(feats.unsqueeze(1)).squeeze(1)
        feats = self.drop(feats)
        return self.fc(feats).squeeze(-1)


class ClassHead(nn.Module):
    def __init__(self, feature_dim=256, num_classes=80, dropout_p=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class GRG(nn.Module):
    """
    Global Relation Graph (GRG)-style multi-head GAT
    """

    def __init__(self, feature_dim=256, num_heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            in_dim = feature_dim
            out_dim = feature_dim // num_heads
            gat = GATConv(
                in_dim, out_dim, heads=num_heads, concat=True, dropout=dropout
            )
            self.gat_layers.append(gat)

        self.norms = nn.ModuleList(
            [nn.LayerNorm(feature_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout * 0.5)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, pos_embed=None, scale_embed=None):
        N = x.size(0)
        device = x.device

        idx = torch.arange(N, device=device)
        edge_index = torch.combinations(idx, r=2).t().to(device)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        if pos_embed is not None:
            x = x + pos_embed
        if scale_embed is not None:
            x = x + scale_embed

        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
            x = norm(x + residual)

        return x


class TSPNet(nn.Module):
    def __init__(
        self,
        salient_region_extractor=None,
        feature_dim=256,
        attn_heads=4,
        threshold=0.5,
        dropout_p=0.2,
        gat_k=8,
        **kwargs,
    ):
        super().__init__()
        _ = threshold, gat_k, kwargs
        self.salient_region_extractor = salient_region_extractor
        if self.salient_region_extractor is not None:
            for p in self.salient_region_extractor.parameters():
                p.requires_grad = True
        self.feature_dim = feature_dim
        self.batch_embed = nn.Embedding(
            num_embeddings=8, embedding_dim=self.feature_dim
        )
        self.gat = GRG(feature_dim=self.feature_dim, num_heads=4, num_layers=4)
        self.rank_head = TransformerRankHead(
            feature_dim=feature_dim, dropout_p=dropout_p
        )

    @staticmethod
    def _is_image_input(x: torch.Tensor) -> bool:
        return x.dim() == 4 and x.size(1) == 3

    @staticmethod
    def _normalize_obj_feats(obj_feats: torch.Tensor) -> torch.Tensor:
        if obj_feats.dim() == 3:
            return obj_feats.reshape(-1, obj_feats.shape[-1])
        if obj_feats.dim() != 2:
            raise ValueError(
                f"Expected obj_feats shape [N, D] or [B, N, D], got {tuple(obj_feats.shape)}"
            )
        return obj_feats

    def _apply_gat_per_image(
        self, obj_feats: torch.Tensor, batch_idx: torch.Tensor, num_batches: int
    ) -> torch.Tensor:
        fused_feats = obj_feats.clone()
        for b in range(num_batches):
            m = batch_idx == b
            if not torch.any(m):
                continue
            fused_feats[m] = self.gat(obj_feats[m], pos_embed=None)
        return fused_feats

    def _rank_from_obj_feats(
        self, obj_feats: torch.Tensor, rois: torch.Tensor | None
    ) -> dict:
        """GAT + rank head on precomputed ROI features (component-analysis path)."""
        obj_feats = self._normalize_obj_feats(obj_feats)
        if obj_feats.size(-1) != self.feature_dim:
            raise ValueError(
                f"obj_feats last dim {obj_feats.size(-1)} != feature_dim {self.feature_dim}"
            )

        if rois is not None and rois.numel() > 0:
            if rois.dim() != 2 or rois.size(-1) != 5:
                raise ValueError(f"rois must be [N_rois, 5], got {tuple(rois.shape)}")
            rois = rois.to(obj_feats.device)
            batch_idx = rois[:, 0].long()
            num_batches = int(batch_idx.max().item()) + 1
            fused_feats = self._apply_gat_per_image(
                obj_feats, batch_idx, num_batches
            )
        else:
            batch_idx = torch.zeros(
                obj_feats.size(0), dtype=torch.long, device=obj_feats.device
            )
            num_batches = 1
            fused_feats = self.gat(obj_feats, pos_embed=None)

        rank_scores = self.rank_head(fused_feats).reshape(-1)
        ranks_per_image = [
            rank_scores[batch_idx == b] for b in range(num_batches)
        ]
        phrase_saliency_score_flat = torch.zeros_like(rank_scores)
        phrase_scores_per_image = [
            phrase_saliency_score_flat[batch_idx == b] for b in range(num_batches)
        ]
        zero_loss = rank_scores.sum() * 0.0

        return {
            "rank_score": ranks_per_image,
            "rank_score_flat": rank_scores,
            "phrase_overlay_loss": zero_loss,
            "phrase_saliency_score": phrase_scores_per_image,
            "phrase_saliency_score_flat": phrase_saliency_score_flat,
            "class_logits": None,
            "mask": None,
        }

    def _forward_integrated(
        self, x: torch.Tensor, rois: torch.Tensor, phrases
    ) -> dict:
        B, _, _, _ = x.shape
        batch_idx = rois[:, 0].long()

        obj_feats, pred_mask, class_logits, phrase_saliency_scores = (
            self.salient_region_extractor(x, rois, phrases)
        )
        batch_idx = rois[:, 0].long()
        fused_feats = self._apply_gat_per_image(obj_feats, batch_idx, B)
        rank_scores = self.rank_head(fused_feats).reshape(-1)

        ranks_per_image = [rank_scores[batch_idx == b] for b in range(B)]
        classes_per_image = [class_logits[batch_idx == b] for b in range(B)]
        phrase_scores_per_image = [
            phrase_saliency_scores[batch_idx == b].reshape(-1) for b in range(B)
        ]

        return {
            "rank_score": ranks_per_image,
            "class_logits": classes_per_image,
            "mask": pred_mask,
            "phrase_saliency_score": phrase_scores_per_image,
            "phrase_saliency_score_flat": phrase_saliency_scores.reshape(-1),
        }

    def forward(self, x, rois=None, phrases=None, **kwargs):
        del kwargs
        if self._is_image_input(x):
            if self.salient_region_extractor is None:
                raise ValueError(
                    "MSRModel received images [B, 3, H, W] but salient_region_extractor "
                    "is None. Pass precomputed ROI features [N, D] or provide an extractor."
                )
            if rois is None:
                raise ValueError("rois are required for integrated MSR forward.")
            return self._forward_integrated(x, rois, phrases)
        return self._rank_from_obj_feats(x, rois)
