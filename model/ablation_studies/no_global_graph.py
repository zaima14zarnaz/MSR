"""
Model description: Same as variant E but with a class head, and NO global graph (GRG).
- Removed GRG module + per-image GAT reasoning.
- Salient region extractor features go directly to the rank head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRankHead(nn.Module):
    def __init__(self, feature_dim, num_heads=2, depth=4, dropout_p=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dropout=dropout_p,
            activation="gelu",
            batch_first=False,  # keep your original behavior (seq_len, batch, dim)
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.drop = nn.Dropout(dropout_p)
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, feats):
        # feats: [N, D]
        feats = self.tr(feats.unsqueeze(1)).squeeze(1)  # [N, D]
        feats = self.drop(feats)
        return self.fc(feats).squeeze(-1)  # [N]


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
        # x: [N_rois, feature_dim]
        return self.classifier(x)  # [N_rois, num_classes]


class LGSRModel(nn.Module):
    """
    Variant: No Global Relation Graph (GRG)
    --------------------------------------
    - ROI features from salient_region_extractor go straight to rank_head.
    - Keeps class logits output from salient_region_extractor (class head).
    """

    def __init__(self, salient_region_extractor, feature_dim=256, dropout_p=0.2):
        super().__init__()
        self.salient_region_extractor = salient_region_extractor
        for p in self.salient_region_extractor.parameters():
            p.requires_grad = True

        self.feature_dim = feature_dim
        self.rank_head = TransformerRankHead(feature_dim=feature_dim, dropout_p=dropout_p)

    def forward(self, x, rois, phrases=None):
        """
        Args:
            x: [B, C, H, W]
            rois: [N_rois, 5] where rois[:,0] is batch index
            phrases: optional (kept for API compatibility; not used unless your extractor uses it)
        Returns:
            dict with per-image rank scores + class logits + mask
        """
        B, C, H, W = x.shape
        batch_idx = rois[:, 0].long()

        # 1) ROI -> features
        # If your extractor supports phrases, swap to:
        # obj_feats, pred_mask, class_logits = self.salient_region_extractor(x, rois, phrases=phrases)
        obj_feats, pred_mask, class_logits = self.salient_region_extractor(x, rois)

        # 2) Direct ranking (NO GRG)
        rank_scores = self.rank_head(obj_feats)  # [N_rois]

        # 3) Split per image
        ranks_per_image = [rank_scores[batch_idx == b] for b in range(B)]
        classes_per_image = [class_logits[batch_idx == b] for b in range(B)]

        return {
            "rank_score": ranks_per_image,
            "class_logits": classes_per_image,
            "mask": pred_mask,
        }
