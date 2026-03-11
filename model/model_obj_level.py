import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchvision import models

from scale_aggregators.weighted_scale_aggregator import WeightedScaleAggregator
from scale_aggregators.cross_att_aggregator import CrossAttentionAggregator
from scale_aggregators.fpn_aggregator import FPNAggregator
from scale_aggregators.deconvolution_aggregator import DeconvAggregator
from scale_aggregators.pixel_shuffle_aggregator import PixelShuffleAggregator
from scale_aggregators.qagnet_aggregator import HierarchicalGraphAggregator
from graph_relationship_module import GATModule
from graph_relationship_module import RelationshipModule
from graph_relationship_module import LightweightRel

from positional_encoding import build_2d_sincos_pos_embed
from positional_encoding import roi_to_xywh

# # class MultiscaleDeconvAggrRelModSRNet(nn.Module):
# #     """
# #     Object-level Relationship Reasoning Network using Pretrained Saliency Features
# #     -------------------------------------------------------------------------------
# #     - Uses pretrained MultiScaleSaliency model to extract per-object features.
# #     - Pools saliency-aware multi-scale features within object masks.
# #     - Aggregates geometric + appearance features for relational reasoning and ranking.
# #     """

# #     def __init__(self, pretrained_saliency_model, feature_dim=256, attn_heads=4, threshold=0.5, dropout_p=0.2):
# #         super().__init__()
# #         self.saliency_model = pretrained_saliency_model
# #         for p in self.saliency_model.parameters():
# #             p.requires_grad = True  # fine-tune allowed
# #         self.feature_dim = feature_dim

# #         self.pos_proj = nn.Linear(feature_dim, feature_dim)
# #         self.pos_drop = nn.Dropout(dropout_p)
# #         self.pos_scale = nn.Parameter(torch.ones(1))  # optional

# #         # inside __init__ of MultiscaleDeconvAggrRelModSRNet
# #         self.batch_embed = nn.Embedding(num_embeddings=16, embedding_dim=self.feature_dim)

# #         self.rank_head = nn.Sequential(
# #             nn.Linear(self.feature_dim, 64),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(dropout_p * 2.0),
# #             nn.Linear(64, 32),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(dropout_p * 1.5),
# #             nn.Linear(32, 1)
# #         )


# #     # def forward(self, x, rois, obj_masks=None):
# #     #     B, C, H, W = x.shape
# #     #     batch_idx = rois[:, 0].long()

# #     #     # 1️⃣ ROI → features
# #     #     obj_feats = self.saliency_model(x, rois)  # [total_rois, C]

# #     #     # 2️⃣ Rank scores for all ROIs (no positional info)
# #     #     rank_scores = self.rank_head(obj_feats).squeeze(-1)  # [total_rois]

# #     #     # 3️⃣ Group scores per image
# #     #     ranks_per_image = [rank_scores[batch_idx == b] for b in range(B)]

# #     #     return {'rank_score': ranks_per_image}

# #     def forward(self, x, rois, obj_masks=None):
# #         B, C, H, W = x.shape
# #         batch_idx = rois[:, 0].long()

# #         # 1️⃣ ROI → features
# #         obj_feats = self.saliency_model(x, rois)  # [total_rois, C]

# #         # 2️⃣ ROI → positional encoding (spatial + batch)
# #         rois_xywh = roi_to_xywh(rois)  # [total_rois, 4]
# #         pos_embed = build_2d_sincos_pos_embed(
# #             rois_xywh, self.feature_dim, img_size=(H, W)
# #         ).to(obj_feats.device)  # [total_rois, feature_dim]

# #         # add per-image embedding to distinguish batches
# #         if hasattr(self, "batch_embed"):
# #             pos_embed = pos_embed + self.batch_embed(batch_idx)  # [total_rois, feature_dim]

# #         # project + dropout (optional)
# #         if hasattr(self, "pos_proj"):
# #             pos_embed = self.pos_proj(pos_embed)
# #         if hasattr(self, "pos_drop"):
# #             pos_embed = self.pos_drop(pos_embed)

# #         # add position info to ROI features
# #         obj_feats = obj_feats + pos_embed

# #         # 3️⃣ Rank scores for all ROIs
# #         rank_scores = self.rank_head(obj_feats).squeeze(-1)  # [total_rois]

# #         # 4️⃣ Group scores per image
# #         ranks_per_image = [rank_scores[batch_idx == b] for b in range(B)]

# #         return {'rank_score': ranks_per_image}
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiscaleDeconvAggrRelModSRNet(nn.Module):
#     """
#     Object-level Relationship Reasoning Network with Cross-Attention
#     ----------------------------------------------------------------
#     - Uses pretrained MultiScaleSaliency model to extract per-object features.
#     - Adds positional encodings and cross-attention between ROIs.
#     - Outputs per-object saliency rank scores.
#     """

#     def __init__(self, pretrained_saliency_model, feature_dim=256, attn_heads=4, threshold=0.5, dropout_p=0.2):
#         super().__init__()
#         self.saliency_model = pretrained_saliency_model
#         for p in self.saliency_model.parameters():
#             p.requires_grad = True

#         self.feature_dim = feature_dim

#         # Positional embedding projection and scaling
#         self.pos_proj = nn.Linear(feature_dim, feature_dim)
#         self.pos_drop = nn.Dropout(dropout_p)
#         self.pos_scale = nn.Parameter(torch.ones(1))

#         # Batch embedding for image-level distinction
#         self.batch_embed = nn.Embedding(num_embeddings=16, embedding_dim=feature_dim)

#         # Cross-attention block (self-attention among ROIs per image)
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=feature_dim,
#             num_heads=attn_heads,
#             dropout=dropout_p,
#             batch_first=True  # [N, L, C] input format
#         )

#         # Rank head
#         self.rank_head = nn.Sequential(
#             nn.Linear(feature_dim, 64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p * 2.0),
#             nn.Linear(64, 32),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p * 1.5),
#             nn.Linear(32, 1)
#         )

#     def forward(self, x, rois, obj_masks=None):
#         B, C, H, W = x.shape
#         batch_idx = rois[:, 0].long()

#         # 1️⃣ ROI → appearance features
#         obj_feats = self.saliency_model(x, rois)  # [total_rois, C]

#         # 2️⃣ ROI → positional encoding (spatial + batch)
#         rois_xywh = roi_to_xywh(rois)  # [total_rois, 4]
#         pos_embed = build_2d_sincos_pos_embed(rois_xywh, self.feature_dim, img_size=(H, W)).to(obj_feats.device)

#         pos_embed = pos_embed + self.batch_embed(batch_idx)
#         pos_embed = self.pos_proj(pos_embed)
#         pos_embed = self.pos_drop(pos_embed)

#         # Add scaled positional info
#         obj_feats = obj_feats + self.pos_scale * pos_embed

#         # 3️⃣ Cross-Attention per image
#         attn_outs = []
#         for b in range(B):
#             mask = batch_idx == b
#             feats_b = obj_feats[mask]  # [num_rois_in_b, C]
#             if feats_b.size(0) == 0:
#                 continue
#             # Self-attention: query, key, value all = feats_b
#             attn_out, _ = self.cross_attn(feats_b.unsqueeze(0), feats_b.unsqueeze(0), feats_b.unsqueeze(0))
#             attn_outs.append(attn_out.squeeze(0))

#         # Concatenate back all ROI features (same order as input)
#         attn_feats = torch.cat(attn_outs, dim=0)

#         # 4️⃣ Rank scores
#         rank_scores = self.rank_head(attn_feats).squeeze(-1)

#         # 5️⃣ Group scores per image
#         ranks_per_image = [rank_scores[batch_idx == b] for b in range(B)]

#         return {'rank_score': ranks_per_image}



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

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


class MultiscaleDeconvAggrRelModSRNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=256, attn_heads=4, threshold=0.5, dropout_p=0.2, gat_k=8):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True
        self.feature_dim = feature_dim
        self.gat_k = gat_k
        self.pos_proj = nn.Linear(feature_dim, feature_dim)
        self.pos_drop = nn.Dropout(dropout_p)
        self.pos_scale = nn.Parameter(torch.ones(1))
        self.batch_embed = nn.Embedding(num_embeddings=16, embedding_dim=self.feature_dim)
        self.gat = GRG(feature_dim=self.feature_dim, num_heads=4, num_layers=4)

        self.rank_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 2.0),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 1.5),
            nn.Linear(32, 1)
        )

    def forward(self, x, rois, obj_masks=None):
        B, C, H, W = x.shape
        batch_idx = rois[:, 0].long()

        # 1️⃣ ROI → features
        obj_feats = self.saliency_model(x, rois)  # [total_rois, C]


        # keep a copy of original features for residual fusion
        raw_feats = obj_feats.clone()

        # 3️⃣ Per-image GAT message passing
        outs = []
        for b in range(B):
            m = batch_idx == b
            if not torch.any(m):
                continue
            feats_b = obj_feats[m]

            # global GRG-style reasoning
            feats_b = self.gat(feats_b, pos_embed=None)
            outs.append((m, feats_b))


        # 4️⃣ Merge all back into one tensor
        fused_feats = raw_feats.clone()
        for m, f in outs:
            fused_feats[m] = f

        # 5️⃣ Rank prediction
        rank_scores = self.rank_head(fused_feats).squeeze(-1)

        # 6️⃣ Group per image
        ranks_per_image = [rank_scores[batch_idx == b] for b in range(B)]

        return {'rank_score': ranks_per_image}




