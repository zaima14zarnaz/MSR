"""
Model description: Same as variant E but with a class head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchvision import models

from positional_encoding import build_2d_sincos_pos_embed
from positional_encoding import roi_to_xywh


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import torch
import torch.nn as nn



class TransformerRankHead(nn.Module):
    def __init__(self, feature_dim, num_heads=2, depth=4, dropout_p=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dropout=dropout_p,          # dropout inside attention and FFN
            activation='gelu'
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.drop = nn.Dropout(dropout_p)  # extra dropout before final layer
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, feats):
        # feats: [N, D]
        feats = self.tr(feats.unsqueeze(1)).squeeze(1)
        feats = self.drop(feats)  # regularization before prediction
        return self.fc(feats).squeeze(-1)

class ClassHead(nn.Module):
    def __init__(self, feature_dim=256, num_classes=80, dropout_p=0.2):
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
        return self.classifier(x)  # [N_rois, num_classes]


class GRG(nn.Module):
    """
    Global Relation Graph (GRG)-style multi-head GAT
    -------------------------------------------------
    - Fully connected graph over all ROIs in an image.
    - Multi-head attention.
    - Residual connections and layer normalization for stability.
    - Incorporates positional and scale embeddings.
    """

    def __init__(self, feature_dim=256, num_heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = feature_dim
            out_dim = feature_dim // num_heads
            gat = GATConv(in_dim, out_dim, heads=num_heads, concat=True, dropout=dropout)
            self.gat_layers.append(gat)

        self.norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout * 0.5)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, pos_embed=None, scale_embed=None):
        """
        Args:
            x: [N, D] ROI features (already fused from saliency network)
            pos_embed: [N, D] optional positional encoding
            scale_embed: [N, D] optional scale encoding
        Returns:
            x: [N, D] globally contextualized ROI features
        """
        N = x.size(0)
        device = x.device

        # fully connected edges (no kNN masking)
        idx = torch.arange(N, device=device)
        edge_index = torch.combinations(idx, r=2).t().to(device)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected

        # optional: add positional + scale embeddings
        if pos_embed is not None:
            x = x + pos_embed
        if scale_embed is not None:
            x = x + scale_embed

        # pass through stacked GAT layers with residuals
        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
            x = norm(x + residual)

        return x


class LGSRModel(nn.Module):
    def __init__(self, salient_region_extractor, feature_dim=256, attn_heads=4, threshold=0.5, dropout_p=0.2, gat_k=8):
        super().__init__()
        self.salient_region_extractor = salient_region_extractor
        for p in self.salient_region_extractor.parameters():
            p.requires_grad = True
        self.feature_dim = feature_dim
        self.gat_k = gat_k
        self.batch_embed = nn.Embedding(num_embeddings=8, embedding_dim=self.feature_dim)
        self.gat = GRG(feature_dim=self.feature_dim, num_heads=4, num_layers=4)

        self.rank_head = TransformerRankHead(feature_dim=feature_dim, dropout_p=0.2)

    def forward(self, x, rois, phrases=None):
        B, C, H, W = x.shape
        batch_idx = rois[:, 0].long()

        # 1️⃣ ROI → features
        # todo: send phrases to the salient_region_extractor module's forward pass
        obj_feats, pred_mask, class_logits = self.salient_region_extractor(x, rois)  # [total_rois, feature_dim]
        raw_feats = obj_feats.clone()

        # 2️⃣ Per-image GAT reasoning
        outs = []
        for b in range(B):
            m = batch_idx == b
            if not torch.any(m):
                continue
            feats_b = obj_feats[m]
            feats_b = self.gat(feats_b, pos_embed=None)
            outs.append((m, feats_b))

        # 3️⃣ Merge refined feats
        fused_feats = raw_feats.clone()
        for m, f in outs:
            fused_feats[m] = f

        # 4️⃣ Predictions
        rank_scores = self.rank_head(fused_feats).squeeze(-1)    # [N_rois]

        # 5️⃣ Split per image
        ranks_per_image = [rank_scores[batch_idx == b] for b in range(B)]
        classes_per_image = [class_logits[batch_idx == b] for b in range(B)]

        return {
            "rank_score": ranks_per_image,
            "class_logits": classes_per_image,
            "mask": pred_mask
        }




