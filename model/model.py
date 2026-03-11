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


# class GATModule(nn.Module):
#     def __init__(self, in_dim=256, hidden_dim=128, out_dim=256, heads=4):
#         super().__init__()
#         self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
#         self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)
#         self.norm = nn.LayerNorm(out_dim)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # x: [B, C, H, W]
#         B, C, H, W = x.shape
#         x_flat = x.flatten(2).permute(0, 2, 1)  # [B, N, C] where N = H*W
#         N = H * W

#         # Fully connected graph adjacency (dense graph)
#         edge_index = torch.combinations(torch.arange(N, device=x.device), r=2).t()
#         edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # bidirectional

#         out_feats = []
#         for b in range(B):
#             feat = x_flat[b]  # [N, C]
#             feat = self.gat1(feat, edge_index)
#             feat = self.act(feat)
#             feat = self.gat2(feat, edge_index)
#             feat = self.norm(feat)
#             feat = self.act(feat)
#             out_feats.append(feat)

#         out_feats = torch.stack(out_feats, dim=0)  # [B, N, out_dim]
#         out_feats = out_feats.permute(0, 2, 1).view(B, out_feats.shape[-1], H, W)
#         return out_feats

class AggregatedSaliencyRankNet(nn.Module):
    def __init__(self, pretrained_saliency_model):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = False

        self.aggregator = WeightedScaleAggregator(feature_dim=256, n_scales=3)

        # Replace CNN with GAT
        self.gat = GATModule(in_dim=256, hidden_dim=128, out_dim=256, heads=4)

        self.rank_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x, obj_masks=None):
        with torch.no_grad():
            out = self.saliency_model(x)

        F_list = out['F_per_scale']
        logits_list = out['logits_per_scale']
        A_scales = out['A_per_scale']
        w_scales = out['w_scales']

        salient_F_list = []
        for F_i, logit_i in zip(F_list, logits_list):
            sal_mask = torch.sigmoid(logit_i)
            sal_mask = F.interpolate(sal_mask, size=F_i.shape[-2:], mode='bilinear', align_corners=False)
            F_salient = F_i * sal_mask
            salient_F_list.append(F_salient)

        agg_feat = self.aggregator(salient_F_list, w_scales, A_scales)
        agg_feat = self.gat(agg_feat)  # GAT refinement step

        if obj_masks is None:
            pooled_feat = F.adaptive_avg_pool2d(agg_feat, 1).flatten(1)
            rank_scores = self.rank_head(pooled_feat)
        else:
            batch_rank_scores = []
            for i in range(len(obj_masks)):
                masks = obj_masks[i].to(agg_feat.device)
                feats = agg_feat[i]

                obj_feats = []
                for j in range(masks.shape[0]):
                    mask = masks[j].unsqueeze(0).unsqueeze(0)
                    mask_resized = F.interpolate(mask, size=feats.shape[-2:], mode='bilinear', align_corners=False)
                    mask_resized = mask_resized.squeeze(0)
                    area = mask_resized.sum() + 1e-6
                    pooled = (feats * mask_resized).sum(dim=(1,2)) / area
                    obj_feats.append(pooled)

                obj_feats = torch.stack(obj_feats, dim=0)
                obj_scores = self.rank_head(obj_feats).squeeze(-1)
                batch_rank_scores.append(obj_scores)

            rank_scores = batch_rank_scores

        return {
            'aggregated_feature': agg_feat,
            'rank_score': rank_scores,
            'logits_final': out['logits_final']
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# without geometric features
class SimplifiedSaliencyRankNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=2048, attn_heads=4, threshold=0.5):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True

        self.aggregator = WeightedScaleAggregator(feature_dim=256, n_scales=3)

        # Use ResNet152 pretrained backbone
        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # until last conv layer
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.threshold = threshold

        # --- Cross-attention: fuse saliency + CNN features ---
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=attn_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(feature_dim)

        # --- Self-attention across object-level features ---
        self.self_norm1 = nn.LayerNorm(feature_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=attn_heads, batch_first=True)
        self.self_norm2 = nn.LayerNorm(feature_dim)

        # --- Rank prediction head (MLP) ---
        self.rank_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        for module in [self.rank_head, self.self_attn, self.cross_attn]:
            for p in module.parameters():
                p.requires_grad = True


    def forward(self, x, obj_masks=None):
        sal_out = self.saliency_model(x)
        F_list = sal_out['F_per_scale']
        logits_list = sal_out['logits_per_scale']
        A_scales = sal_out['A_per_scale']
        w_scales = sal_out['w_scales']
        F_last_resnet = self.backbone(x)  # [B, 2048, H', W']

        # --- Saliency-masked CNN features ---
        saliency_map = F_last_resnet.pow(2).mean(dim=1, keepdim=True)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)
        sal_mask_bin = (saliency_map > self.threshold).float()
        salient_feats = F_last_resnet * sal_mask_bin  # [B, 2048, H', W']

        # --- Saliency-based aggregation ---
        salient_F_list = []
        for F_i, logit_i in zip(F_list, logits_list):
            sal_mask = torch.sigmoid(logit_i)
            sal_mask = F.interpolate(sal_mask, size=F_i.shape[-2:], mode='bilinear', align_corners=False)
            F_salient = F_i * sal_mask
            salient_F_list.append(F_salient)
        agg_feat = self.aggregator(salient_F_list, w_scales, A_scales)  # [B,256,H',W']

        # --- Project agg_feat to same dim (2048) for attention matching ---
        if agg_feat.shape[1] != salient_feats.shape[1]:
            proj = nn.Conv2d(agg_feat.shape[1], salient_feats.shape[1], kernel_size=1).to(agg_feat.device)
            agg_feat = proj(agg_feat)

        # --- Cross-attention: fuse multi-scale (agg_feat) with global (salient_feats) ---
        B, C, H, W = salient_feats.shape
        query = salient_feats.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        key = agg_feat.flatten(2).permute(0, 2, 1)         # [B, HW, C]
        value = key

        attended, _ = self.cross_attn(query, key, value)
        fused_feats = self.cross_norm(attended + query)     # residual fusion
        fused_feats = fused_feats.permute(0, 2, 1).reshape(B, C, H, W)  # back to [B,C,H,W]

        # --- Object-level pooling ---
        obj_feats_batch = []
        if obj_masks is not None:
            for i in range(B):
                feats = fused_feats[i]
                masks = obj_masks[i].to(feats.device)
                masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
                obj_feats = []
                for m in masks:
                    area = m.sum() + 1e-6
                    pooled = (feats * m).sum(dim=(1, 2)) / area
                    obj_feats.append(pooled)
                obj_feats = torch.stack(obj_feats, dim=0)
                obj_feats_batch.append(obj_feats)
        else:
            for i in range(B):
                feats = fused_feats[i].flatten(1)
                mask = sal_mask_bin[i].flatten(1)
                valid = mask.squeeze(0) > 0
                if valid.sum() < 1:
                    pooled = feats.mean(dim=1, keepdim=True).t()
                    obj_feats_batch.append(pooled)
                    continue
                selected = feats[:, valid].t()
                obj_feats_batch.append(selected)

        # --- Self-attention across object embeddings + rank prediction ---
        rank_outputs = []
        for feats in obj_feats_batch:
            feats = self.self_norm1(feats)
            if feats.shape[0] > 1:
                attended, _ = self.self_attn(feats, feats, feats)
                feats = self.self_norm2(attended)
            ranks = self.rank_head(feats).squeeze(-1)
            rank_outputs.append(ranks)

        return {
            'rank_score': rank_outputs,
            'cross_attended_features': fused_feats,
            'saliency_mask': sal_mask_bin,
            'agg_feat': agg_feat
        }


# with geometric features:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# With multiscale features aggregated using an aggregator
class MultiscaleAggregationSRNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=2048, attn_heads=4, threshold=0.5):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True

        self.proj = nn.Conv2d(256, 2048, kernel_size=1)
        # self.aggregator = WeightedScaleAggregator(feature_dim=256, n_scales=3)
        # self.aggregator = CrossAttentionAggregator(feature_dim=256, n_scales=3)
        self.aggregator = FPNAggregator(feature_dim=256, n_scales=3)

        # --- Backbone (ResNet152) ---
        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.threshold = threshold

        self.embed_dim = (feature_dim + 6)
        self.attn_embed_dim = (self.embed_dim // attn_heads) * attn_heads  # nearest divisible
        self.pre_attn_proj = nn.Linear(self.embed_dim, self.attn_embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=self.attn_embed_dim, num_heads=attn_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(self.attn_embed_dim)

        self.self_norm1 = nn.LayerNorm(self.attn_embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=self.attn_embed_dim, num_heads=attn_heads, batch_first=True)
        self.self_norm2 = nn.LayerNorm(self.attn_embed_dim)
        # --- Rank head ---
        self.rank_head = nn.Sequential(
            nn.Linear(self.attn_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )


    # =============================================================
    # Geometric feature extractor
    # =============================================================
    def build_geometric_features(self, obj_masks, H, W, device):
        geom_feats_batch = []
        for masks in obj_masks:
            geom_feats = []
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W),
                                  mode='bilinear', align_corners=False).squeeze(1)
            for m in masks:
                y_idx, x_idx = torch.nonzero(m > 0.5, as_tuple=True)
                if len(x_idx) == 0:
                    geom_feats.append(torch.zeros(6, device=device))
                    continue
                xmin, xmax = x_idx.min().float(), x_idx.max().float()
                ymin, ymax = y_idx.min().float(), y_idx.max().float()
                w_obj = (xmax - xmin + 1) / W
                h_obj = (ymax - ymin + 1) / H
                x_c = (xmin + xmax + 1) / (2 * W)
                y_c = (ymin + ymax + 1) / (2 * H)
                area_norm = w_obj * h_obj
                aspect = w_obj / (h_obj + 1e-6)
                geom_feats.append(torch.tensor([x_c, y_c, w_obj, h_obj, area_norm, aspect], device=device))
            geom_feats = torch.stack(geom_feats, dim=0)
            geom_feats_batch.append(geom_feats)
        return geom_feats_batch

    # =============================================================
    # Forward pass
    # =============================================================
    def forward(self, x, bin_gt, obj_masks=None):
        # --- Saliency model forward ---
        sal_out = self.saliency_model(x, bin_gt)
        F_list = sal_out['F_per_scale']
        logits_list = sal_out['logits_per_scale']
        A_scales = sal_out['A_per_scale']
        w_scales = sal_out['w_scales']
        bin_loss = sal_out['loss']

        with torch.no_grad():
            F_last_resnet = self.backbone(x)  # [B, 2048, H', W']

        # --- Saliency-masked CNN features ---
        saliency_map = F_last_resnet.pow(2).mean(dim=1, keepdim=True)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)
        sal_mask_bin = (saliency_map > self.threshold).float()
        salient_feats = F_last_resnet * sal_mask_bin

        # --- Aggregated saliency feature ---
        salient_F_list = []
        for F_i, logit_i in zip(F_list, logits_list):
            sal_mask = torch.sigmoid(logit_i)
            sal_mask = F.interpolate(sal_mask, size=F_i.shape[-2:], mode='bilinear', align_corners=False)
            F_salient = F_i * sal_mask
            salient_F_list.append(F_salient)
        agg_feat = self.aggregator(salient_F_list, w_scales, A_scales=None)
        if agg_feat.shape[1] != salient_feats.shape[1]:
            agg_feat = self.proj(agg_feat)

        # --- Object-level pooling ---
        B, C, H, W = salient_feats.shape
        obj_feats_batch = []
        geom_feats_batch = []
        if obj_masks is not None:
            geom_feats_batch = self.build_geometric_features(obj_masks, H, W, salient_feats.device)
            for i in range(B):
                feats = salient_feats[i]
                masks = obj_masks[i].to(feats.device)
                masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
                obj_feats = []
                for m in masks:
                    area = m.sum() + 1e-6
                    pooled = (feats * m).sum(dim=(1, 2)) / area
                    obj_feats.append(pooled)
                obj_feats = torch.stack(obj_feats, dim=0)
                # concatenate geometry before attention
                obj_feats = torch.cat([obj_feats, geom_feats_batch[i]], dim=-1)
                obj_feats_batch.append(obj_feats)
        else:
            obj_feats_batch = [salient_feats.mean(dim=(2, 3)).unsqueeze(1)]

        # --- Cross-attention between object embeddings ---
        rank_outputs = []
        for feats in obj_feats_batch:
            feats = self.pre_attn_proj(feats)      # project 2054 → 2052 first
            feats = self.self_norm1(feats)         # now matches LayerNorm
            if feats.shape[0] > 1:
                attended, _ = self.cross_attn(feats, feats, feats)
                feats = self.cross_norm(attended + feats)
            feats = self.self_norm2(feats)
            ranks = self.rank_head(feats).squeeze(-1)
            rank_outputs.append(ranks)

        return {
            'rank_score': rank_outputs,
            'saliency_mask': sal_mask_bin,
            'agg_feat': agg_feat,
            'bin_loss': bin_loss
        }

# Same as previous one, but the cross attention block is replaced with a Simple GAT message passing layer
class MultiscaleAggrGATSRNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=2048, attn_heads=4, threshold=0.5, dropout_p=0.1):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True

        # small conv regularization for projection
        self.proj = nn.Sequential(
            nn.Conv2d(256, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),          # <--- batch norm for stability
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)        # <--- dropout in feature space
        )

        self.aggregator = FPNAggregator(feature_dim=256, n_scales=3)

        # frozen backbone (to prevent overfitting on small data)
        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.threshold = threshold
        self.embed_dim = feature_dim + 6
        self.attn_embed_dim = (self.embed_dim // attn_heads) * attn_heads
        self.pre_attn_proj = nn.Linear(self.embed_dim, self.attn_embed_dim)

        # --- GAT with dropout ---
        self.gat = GATModule(
            in_dim=self.attn_embed_dim,
            out_dim=self.attn_embed_dim,
            neg_slope=0.2,
            dropout=dropout_p  # <--- added dropout inside GAT
        )
        self.cross_norm = nn.LayerNorm(self.attn_embed_dim)
        self.self_norm2 = nn.LayerNorm(self.attn_embed_dim)

        # --- Rank head (regularized) ---
        self.rank_head = nn.Sequential(
            nn.Linear(self.attn_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),         # <--- dropout added here
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p / 2),     # <--- smaller dropout for stability
            nn.Linear(128, 1)
        )

    def build_geometric_features(self, obj_masks, H, W, device):
        geom_feats_batch = []
        for masks in obj_masks:
            geom_feats = []
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W),
                                  mode='bilinear', align_corners=False).squeeze(1)
            for m in masks:
                y_idx, x_idx = torch.nonzero(m > 0.5, as_tuple=True)
                if len(x_idx) == 0:
                    geom_feats.append(torch.zeros(6, device=device))
                    continue
                xmin, xmax = x_idx.min().float(), x_idx.max().float()
                ymin, ymax = y_idx.min().float(), y_idx.max().float()
                w_obj = (xmax - xmin + 1) / W
                h_obj = (ymax - ymin + 1) / H
                x_c = (xmin + xmax + 1) / (2 * W)
                y_c = (ymin + ymax + 1) / (2 * H)
                area_norm = w_obj * h_obj
                aspect = w_obj / (h_obj + 1e-6)
                geom_feats.append(torch.tensor([x_c, y_c, w_obj, h_obj, area_norm, aspect], device=device))
            geom_feats_batch.append(torch.stack(geom_feats, dim=0))
        return geom_feats_batch

    def forward(self, x, bin_gt, obj_masks=None):
        sal_out = self.saliency_model(x, bin_gt)
        F_list = sal_out['F_per_scale']
        logits_list = sal_out['logits_per_scale']
        A_scales = sal_out['A_per_scale']
        w_scales = sal_out['w_scales']
        bin_loss = sal_out['loss']

        with torch.no_grad():
            F_last_resnet = self.backbone(x)

        saliency_map = F_last_resnet.pow(2).mean(dim=1, keepdim=True)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)
        sal_mask_bin = (saliency_map > self.threshold).float()
        salient_feats = F_last_resnet * sal_mask_bin

        salient_F_list = []
        for F_i, logit_i in zip(F_list, logits_list):
            sal_mask = torch.sigmoid(logit_i)
            sal_mask = F.interpolate(sal_mask, size=F_i.shape[-2:], mode='bilinear', align_corners=False)
            F_salient = F_i * sal_mask
            salient_F_list.append(F_salient)

        agg_feat = self.aggregator(salient_F_list, w_scales, A_scales=None)
        agg_feat = self.proj(agg_feat)  # <--- now regularized

        B, C, H, W = salient_feats.shape
        obj_feats_batch = []
        if obj_masks is not None:
            geom_feats_batch = self.build_geometric_features(obj_masks, H, W, salient_feats.device)
            for i in range(B):
                feats = salient_feats[i]
                masks = F.interpolate(obj_masks[i].unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
                pooled = [(feats * m).sum(dim=(1, 2)) / (m.sum() + 1e-6) for m in masks]
                obj_feats = torch.stack(pooled, dim=0)
                obj_feats = torch.cat([obj_feats, geom_feats_batch[i]], dim=-1)
                obj_feats_batch.append(obj_feats)
        else:
            obj_feats_batch = [salient_feats.mean(dim=(2, 3)).unsqueeze(1)]

        rank_outputs = []
        for feats in obj_feats_batch:
            feats = self.pre_attn_proj(feats)
            if feats.shape[0] > 1:
                N = feats.shape[0]
                adj = torch.ones(N, N, device=feats.device)
                adj.fill_diagonal_(0)
                attended = self.gat(feats, adj)
                feats = self.cross_norm(attended + feats)
            feats = self.self_norm2(feats)
            ranks = self.rank_head(feats).squeeze(-1)
            rank_outputs.append(ranks)

        return {
            'rank_score': rank_outputs,
            'saliency_mask': sal_mask_bin,
            'agg_feat': agg_feat,
            'bin_loss': bin_loss
        }

# Same as previous one, but the object-object GAT block is replaced with a 
# complex relationship module that finds relationships between the objects
# by computed edge embeddings by a fused representation of source and destination
# nodes, and then message passing between the nodes to create context aware
# object representations
class MultiscaleAggrRelModSRNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=2048, attn_heads=4, threshold=0.5, dropout_p=0.2):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True

        # Project aggregated features to a unified embedding space
        self.proj = nn.Sequential(
            nn.Conv2d(256, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

        # 🔹 Updated FPN aggregator: now supports 4 scales
        self.aggregator = FPNAggregator(feature_dim=256, n_scales=4)

        # --- Frozen backbone (used only for saliency region extraction) ---
        # backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        # self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # self.backbone.eval()
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        self.threshold = threshold
        self.embed_dim = feature_dim + 6
        self.attn_embed_dim = (self.embed_dim // attn_heads) * attn_heads
        self.pre_attn_proj = nn.Linear(self.embed_dim, self.attn_embed_dim)

        # Relationship module (replaces GAT)
        self.relationship_module = RelationshipModule(
            embed_dim=self.attn_embed_dim,
            hidden_dim=512,
            dropout_p=dropout_p,
            num_heads=attn_heads
        )

        self.self_norm2 = nn.LayerNorm(self.attn_embed_dim)

        # --- Rank head for saliency ranking ---
        self.rank_head = nn.Sequential(
            nn.Linear(self.attn_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p / 2),
            nn.Linear(128, 1)
        )

    # --- Build 6D geometric feature vector per object mask ---
    def build_geometric_features(self, obj_masks, H, W, device):
        geom_feats_batch = []
        for masks in obj_masks:
            geom_feats = []
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W),
                                  mode='bilinear', align_corners=False).squeeze(1)
            for m in masks:
                y_idx, x_idx = torch.nonzero(m > 0.5, as_tuple=True)
                if len(x_idx) == 0:
                    geom_feats.append(torch.zeros(6, device=device))
                    continue
                xmin, xmax = x_idx.min().float(), x_idx.max().float()
                ymin, ymax = y_idx.min().float(), y_idx.max().float()
                w_obj = (xmax - xmin + 1) / W
                h_obj = (ymax - ymin + 1) / H
                x_c = (xmin + xmax + 1) / (2 * W)
                y_c = (ymin + ymax + 1) / (2 * H)
                area_norm = w_obj * h_obj
                aspect = w_obj / (h_obj + 1e-6)
                geom_feats.append(torch.tensor([x_c, y_c, w_obj, h_obj, area_norm, aspect], device=device))
            geom_feats_batch.append(torch.stack(geom_feats, dim=0))
        return geom_feats_batch

    # --- Forward Pass ---
    # def forward(self, x, bin_gt, obj_masks=None):
    #     # Get saliency outputs from pretrained binary saliency model
    #     sal_out = self.saliency_model(x, bin_gt)
    #     F_list = sal_out['F_per_scale']      # [F1, F2, F3, F4]
    #     logits_list = sal_out['logits_per_scale']
    #     w_scales = sal_out['w_scales']
    #     # A_scales = sal_out['A_per_scale']
    #     bin_loss = sal_out['loss']

    #     # --- No fusion with backbone feature (cross-attn removed) ---
    #     # The backbone is only used to extract saliency mask
    #     with torch.no_grad():
    #         F_last_resnet = self.backbone(x)

    #     saliency_map = F_last_resnet.pow(2).mean(dim=1, keepdim=True)
    #     saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)
    #     sal_mask_bin = (saliency_map > self.threshold).float()
    #     salient_feats = F_last_resnet * sal_mask_bin

    #     # --- Apply saliency masks to each scale feature ---
    #     salient_F_list = []
    #     for F_i, logit_i in zip(F_list, logits_list):
    #         sal_mask = torch.sigmoid(logit_i)
    #         sal_mask = F.interpolate(sal_mask, size=F_i.shape[-2:], mode='bilinear', align_corners=False)
    #         F_salient = F_i * sal_mask
    #         salient_F_list.append(F_salient)

    #     # --- Aggregate all 4 scales ---
    #     agg_feat = self.aggregator(salient_F_list, w_scales, A_scales=None)
    #     agg_feat = self.proj(agg_feat)  # project to 2048-dim space

    #     # --- Object-wise feature extraction ---
    #     B, C, H, W = agg_feat.shape
    #     obj_feats_batch = []
    #     if obj_masks is not None:
    #         geom_feats_batch = self.build_geometric_features(obj_masks, H, W, agg_feat.device)
    #         for i in range(B):
    #             feats = agg_feat[i]
    #             masks = F.interpolate(obj_masks[i].unsqueeze(1), size=(H, W),
    #                                   mode='bilinear', align_corners=False).squeeze(1)
    #             pooled = [(feats * m).sum(dim=(1, 2)) / (m.sum() + 1e-6) for m in masks]
    #             obj_feats = torch.stack(pooled, dim=0)
    #             obj_feats = torch.cat([obj_feats, geom_feats_batch[i]], dim=-1)
    #             obj_feats_batch.append(obj_feats)
    #     else:
    #         # If no masks, take global pooled features
    #         obj_feats_batch = [agg_feat.mean(dim=(2, 3)).unsqueeze(1)]

    #     # --- Saliency ranking prediction ---
    #     rank_outputs = []
    #     for feats in obj_feats_batch:
    #         feats = self.pre_attn_proj(feats)
    #         if feats.shape[0] > 1:
    #             feats = self.relationship_module(feats)
    #         feats = self.self_norm2(feats)
    #         ranks = self.rank_head(feats).squeeze(-1)
    #         rank_outputs.append(ranks)

    #     return {
    #         'rank_score': rank_outputs,
    #         'saliency_mask': sal_mask_bin,
    #         'agg_feat': agg_feat,
    #         'bin_loss': bin_loss
    #     }
    def forward(self, x, bin_gt, obj_masks=None):
        # --- Step 1: Get saliency model outputs ---
        sal_out = self.saliency_model(x, bin_gt)
        F_list = sal_out['F_per_scale']        # [F1, F2, F3, F4]
        logits_list = sal_out['logits_per_scale']
        w_scales = sal_out['w_scales']
        bin_loss = sal_out['loss']

        # --- Step 2: Derive binary saliency mask from saliency model ---
        # Use final-scale saliency logits for gating
        logits_final = logits_list[-1]
        saliency_map = torch.sigmoid(logits_final)
        sal_mask_bin = (saliency_map > self.threshold).float()

        # --- Step 3: Apply saliency gating to all scale features ---
        salient_F_list = []
        for F_i, logit_i in zip(F_list, logits_list):
            sal_mask = torch.sigmoid(logit_i)
            sal_mask = F.interpolate(sal_mask, size=F_i.shape[-2:], mode='bilinear', align_corners=False)
            F_salient = F_i * sal_mask
            salient_F_list.append(F_salient)

        # --- Step 4: Multi-scale feature aggregation ---
        agg_feat = self.aggregator(salient_F_list, w_scales, A_scales=None)
        agg_feat = self.proj(agg_feat)  # project to unified 2048-d space

        # --- Step 5: Object-level feature extraction ---
        B, C, H, W = agg_feat.shape
        obj_feats_batch = []
        if obj_masks is not None:
            geom_feats_batch = self.build_geometric_features(obj_masks, H, W, agg_feat.device)
            for i in range(B):
                feats = agg_feat[i]
                masks = F.interpolate(obj_masks[i].unsqueeze(1), size=(H, W),
                                    mode='bilinear', align_corners=False).squeeze(1)
                pooled = [(feats * m).sum(dim=(1, 2)) / (m.sum() + 1e-6) for m in masks]
                obj_feats = torch.stack(pooled, dim=0)
                obj_feats = torch.cat([obj_feats, geom_feats_batch[i]], dim=-1)
                obj_feats_batch.append(obj_feats)
        else:
            # Global average pooling if no masks provided
            obj_feats_batch = [agg_feat.mean(dim=(2, 3)).unsqueeze(1)]

        # --- Step 6: Saliency ranking prediction ---
        rank_outputs = []
        for feats in obj_feats_batch:
            feats = self.pre_attn_proj(feats)
            if feats.shape[0] > 1:
                feats = self.relationship_module(feats)
            feats = self.self_norm2(feats)
            ranks = self.rank_head(feats).squeeze(-1)
            rank_outputs.append(ranks)

        # --- Step 7: Return results ---
        return {
            'rank_score': rank_outputs,
            'saliency_mask': sal_mask_bin,  # from saliency model, not backbone
            'agg_feat': agg_feat,
            'bin_loss': bin_loss
        }

# Same as before but the FPN aggregator is replaced with a deconvolution aggregator
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiscaleDeconvAggrRelModSRNet(nn.Module):
    """
    Object-level Relationship Reasoning Network using Pretrained Saliency Features
    -------------------------------------------------------------------------------
    - Uses pretrained MultiScaleSaliency model to extract per-object features.
    - Pools saliency-aware multi-scale features within object masks.
    - Aggregates geometric + appearance features for relational reasoning and ranking.
    """

    def __init__(self, pretrained_saliency_model, feature_dim=256, attn_heads=4, threshold=0.5, dropout_p=0.2):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True  # fine-tune allowed

        self.threshold = threshold
        self.dropout_p = dropout_p

        # --- Feature dimension setup ---
        self.feature_dim = feature_dim
        self.embed_dim = feature_dim + 6  # add geom features
        self.attn_embed_dim = (self.embed_dim // attn_heads) * attn_heads

        # --- Object embedding projection ---
        self.obj_proj = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, self.feature_dim)
        )

        # --- Attention-based relationship reasoning ---
        self.pre_attn_proj = nn.Linear(self.embed_dim, self.attn_embed_dim)
        # self.pre_attn_proj = nn.Linear(256 + 5, 256)


        self.relationship_module = RelationshipModule(
            embed_dim=self.attn_embed_dim,
            hidden_dim=512,
            dropout_p=dropout_p,
            num_heads=attn_heads
        )

        self.self_norm2 = nn.LayerNorm(self.attn_embed_dim)

        # --- Rank head ---
        # self.rank_head = nn.Sequential(
        #     nn.Linear(self.attn_embed_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout_p),
        #     nn.Linear(1024, 1)
        # )

        # --- Rank head (for direct saliency features) ---
        self.rank_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 1)
        )



        # --- Auxiliary image-level supervision ---
        self.aux_rank_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)
        )


    #     return {'rank_score': ranks}
    def forward(self, x, bin_gt=None, obj_masks=None, rois=None):
        # 1️⃣ Extract object-level saliency features
        obj_sal_feats = self.saliency_model(x, rois)  # [num_rois, C]

        # 2️⃣ Project before relationship reasoning
        # feats = self.pre_attn_proj(obj_sal_feats)  # optional projection to attn dimension

        # 3️⃣ Relationship reasoning (self-attention or transformer block)
        # feats = self.relationship_module(feats)
        # feats = self.self_norm2(feats)

        # 4️⃣ Rank prediction
        ranks = self.rank_head(obj_sal_feats).squeeze(-1)

        return {'rank_score': ranks}






import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleAggrSRNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=2048, attn_heads=4, threshold=0.5, dropout_p=0.2):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True

        # Project aggregated features to a unified embedding space
        self.proj = nn.Sequential(
            nn.Conv2d(256, feature_dim, kernel_size=1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

        # Aggregator
        self.aggregator = FPNAggregator(feature_dim=256, n_scales=4)

        # Auxiliary rank supervision (for image-level)
        self.aux_rank_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

        self.threshold = threshold

        # --- Rank head (per object) ---
        self.rank_head = nn.Sequential(
            nn.Linear(feature_dim + 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 1)
        )

    # --- Build geometric features for each object mask ---
    def build_geometric_features(self, obj_masks, H, W, device):
        geom_feats_batch = []
        for masks in obj_masks:
            geom_feats = []
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W),
                                  mode='bilinear', align_corners=False).squeeze(1)
            for m in masks:
                y_idx, x_idx = torch.nonzero(m > 0.5, as_tuple=True)
                if len(x_idx) == 0:
                    geom_feats.append(torch.zeros(6, device=device))
                    continue
                xmin, xmax = x_idx.min().float(), x_idx.max().float()
                ymin, ymax = y_idx.min().float(), y_idx.max().float()
                w_obj = (xmax - xmin + 1) / W
                h_obj = (ymax - ymin + 1) / H
                x_c = (xmin + xmax + 1) / (2 * W)
                y_c = (ymin + ymax + 1) / (2 * H)
                area_norm = w_obj * h_obj
                aspect = w_obj / (h_obj + 1e-6)
                geom_feats.append(torch.tensor([x_c, y_c, w_obj, h_obj, area_norm, aspect], device=device))
            geom_feats_batch.append(torch.stack(geom_feats, dim=0))
        return geom_feats_batch

    def forward(self, x, bin_gt, obj_masks=None):
        # Step 1: get saliency outputs
        sal_out = self.saliency_model(x, bin_gt)
        F_list = sal_out['F_per_scale']
        logits_list = sal_out['logits_per_scale']
        w_scales = sal_out['w_scales']
        bin_loss = sal_out['loss']

        # Step 2: auxiliary image-level rank
        aux_feat = F_list[-1]
        aux_rank_scores = self.aux_rank_head(aux_feat)

        # Step 3: saliency masking per scale
        logits_final = logits_list[-1]
        saliency_map = torch.sigmoid(logits_final)
        sal_mask_bin = (saliency_map > self.threshold).float()

        salient_F_list = []
        for F_i, logit_i in zip(F_list, logits_list):
            sal_mask = torch.sigmoid(logit_i)
            sal_mask = F.interpolate(sal_mask, size=F_i.shape[-2:], mode='bilinear', align_corners=False)
            salient_F_list.append(F_i * sal_mask)

        # Step 4: feature aggregation
        agg_feat = self.aggregator(salient_F_list, w_scales)  # [B, 256, H, W]
        agg_feat = self.proj(agg_feat)                        # [B, 2048, H, W]

        B, C, H, W = agg_feat.shape
        rank_outputs = []

        # Step 5: object-level feature pooling
        if obj_masks is not None:
            geom_feats_batch = self.build_geometric_features(obj_masks, H, W, agg_feat.device)

            for i in range(B):
                feats = agg_feat[i]
                masks = F.interpolate(obj_masks[i].unsqueeze(1), size=(H, W),
                                      mode='bilinear', align_corners=False).squeeze(1)

                pooled = []
                for m in masks:  # each object
                    wsum = m.sum() + 1e-6
                    pooled_feat = (feats * m).sum(dim=(1, 2)) / wsum  # [C]
                    pooled.append(pooled_feat)
                pooled = torch.stack(pooled, dim=0)  # [N_obj, C]

                # concat geometry
                pooled = torch.cat([pooled, geom_feats_batch[i]], dim=-1)  # [N_obj, C+6]

                # predict rank per object
                ranks = self.rank_head(pooled).squeeze(-1)
                rank_outputs.append(ranks)
        else:
            # fallback: single rank per image
            pooled = agg_feat.mean(dim=(2, 3))  # [B, C]
            rank_outputs = [self.rank_head(torch.cat([p, torch.zeros(6, device=p.device)]).unsqueeze(0)).squeeze(-1)
                            for p in pooled]

        return {
            'rank_score': rank_outputs,    # list of [N_obj] tensors (matches training loop)
            'saliency_mask': sal_mask_bin,
            'agg_feat': agg_feat,
            'bin_loss': bin_loss,
            'aux_rank': aux_rank_scores,
        }




# Multiscale features are used but instead of using a weighted aggregator, they are summed up
class MultiscaleSRNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=2048, attn_heads=4, threshold=0.5):
        super().__init__()

        # --- Pretrained Saliency Model (BSD or similar) ---
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True  # fine-tune by default

        # --- Backbone (ResNet152, frozen) ---
        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze backbone

        self.threshold = threshold

        # --- Embedding and Attention Dimensions ---
        self.embed_dim = feature_dim + 6  # +6 for geometric features
        self.attn_embed_dim = (self.embed_dim // attn_heads) * attn_heads  # ensure divisibility
        self.pre_attn_proj = nn.Linear(self.embed_dim, self.attn_embed_dim)

        # --- Attention Modules ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_embed_dim, num_heads=attn_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.attn_embed_dim)

        self.self_norm1 = nn.LayerNorm(self.attn_embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.attn_embed_dim, num_heads=attn_heads, batch_first=True
        )
        self.self_norm2 = nn.LayerNorm(self.attn_embed_dim)

        # --- Ranking Head ---
        self.rank_head = nn.Sequential(
            nn.Linear(self.attn_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        # --- Initialization Note ---
        # for module in [self.rank_head, self.cross_attn, self.self_attn, self.pre_attn_proj]:
        #     for p in module.parameters():
        #         p.requires_grad = True

    # =============================================================
    # Geometric feature extractor
    # =============================================================
    def build_geometric_features(self, obj_masks, H, W, device):
        geom_feats_batch = []
        for masks in obj_masks:
            geom_feats = []
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W),
                                  mode='bilinear', align_corners=False).squeeze(1)
            for m in masks:
                y_idx, x_idx = torch.nonzero(m > 0.5, as_tuple=True)
                if len(x_idx) == 0:
                    geom_feats.append(torch.zeros(6, device=device))
                    continue
                xmin, xmax = x_idx.min().float(), x_idx.max().float()
                ymin, ymax = y_idx.min().float(), y_idx.max().float()
                w_obj = (xmax - xmin + 1) / W
                h_obj = (ymax - ymin + 1) / H
                x_c = (xmin + xmax + 1) / (2 * W)
                y_c = (ymin + ymax + 1) / (2 * H)
                area_norm = w_obj * h_obj
                aspect = w_obj / (h_obj + 1e-6)
                geom_feats.append(torch.tensor([x_c, y_c, w_obj, h_obj, area_norm, aspect], device=device))
            geom_feats = torch.stack(geom_feats, dim=0)
            geom_feats_batch.append(geom_feats)
        return geom_feats_batch


    def forward(self, x, obj_masks=None):
        # --- Saliency model forward ---
        sal_out = self.saliency_model(x)
        logits_list = sal_out['logits_per_scale']

        with torch.no_grad():
            # --- CNN backbone features ---
            F_last_resnet = self.backbone(x)  # [B, 2048, H', W']

        # --- Saliency-masked CNN features ---
        saliency_map = F_last_resnet.pow(2).mean(dim=1, keepdim=True)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)
        sal_mask_bin = (saliency_map > self.threshold).float()
        salient_feats = F_last_resnet * sal_mask_bin  # [B, 2048, H', W']

        # --- ✅ Fixed: Resize logits before combining ---
        # Find target (largest) spatial size
        target_size = salient_feats.shape[-2:]
        resized_logits = [
            F.interpolate(torch.sigmoid(l), size=target_size, mode='bilinear', align_corners=False)
            for l in logits_list
        ]

        # Average resized saliency maps into one guidance mask
        saliency_guidance = torch.mean(torch.stack(resized_logits, dim=0), dim=0)
        salient_feats = salient_feats * saliency_guidance  # reinforce salient regions

        # --- Object-level pooling (unchanged) ---
        B, C, H, W = salient_feats.shape
        obj_feats_batch = []
        if obj_masks is not None:
            geom_feats_batch = self.build_geometric_features(obj_masks, H, W, salient_feats.device)
            for i in range(B):
                feats = salient_feats[i]
                masks = obj_masks[i].to(feats.device)
                masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
                obj_feats = []
                for m in masks:
                    area = m.sum() + 1e-6
                    pooled = (feats * m).sum(dim=(1, 2)) / area
                    obj_feats.append(pooled)
                obj_feats = torch.stack(obj_feats, dim=0)
                obj_feats = torch.cat([obj_feats, geom_feats_batch[i]], dim=-1)
                obj_feats_batch.append(obj_feats)
        else:
            obj_feats_batch = [salient_feats.mean(dim=(2, 3)).unsqueeze(1)]

        # --- Cross-attention between object embeddings ---
        rank_outputs = []
        for feats in obj_feats_batch:
            feats = self.pre_attn_proj(feats)
            feats = self.self_norm1(feats)
            if feats.shape[0] > 1:
                attended, _ = self.cross_attn(feats, feats, feats)
                feats = self.cross_norm(attended + feats)
            feats = self.self_norm2(feats)
            ranks = self.rank_head(feats).squeeze(-1)
            rank_outputs.append(ranks)

        return {
            'rank_score': rank_outputs,
            'saliency_mask': saliency_guidance,  # return the averaged one
            'backbone_feat': F_last_resnet
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Multiscale features are not used, only the final feature map from the backbone netork is used
class SRNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=2048, attn_heads=4, threshold=0.5):
        super().__init__()

        # --- BSD Saliency Model ---
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True  # fine-tune BSD by default

        # --- ResNet152 Backbone (frozen) ---
        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # output: [B, 2048, H/32, W/32]
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.threshold = threshold

        # --- Attention Embedding ---
        self.embed_dim = feature_dim + 6  # visual + geometric
        self.attn_embed_dim = (self.embed_dim // attn_heads) * attn_heads
        self.pre_attn_proj = nn.Linear(self.embed_dim, self.attn_embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_embed_dim, num_heads=attn_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.attn_embed_dim)

        self.self_norm1 = nn.LayerNorm(self.attn_embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.attn_embed_dim, num_heads=attn_heads, batch_first=True
        )
        self.self_norm2 = nn.LayerNorm(self.attn_embed_dim)

        # --- Ranking Head ---
        self.rank_head = nn.Sequential(
            nn.Linear(self.attn_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    # =============================================================
    # Build 6D geometric features
    # =============================================================
    def build_geometric_features(self, obj_masks, H, W, device):
        geom_feats_batch = []
        for masks in obj_masks:
            geom_feats = []
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
            for m in masks:
                y_idx, x_idx = torch.nonzero(m > 0.5, as_tuple=True)
                if len(x_idx) == 0:
                    geom_feats.append(torch.zeros(6, device=device))
                    continue
                xmin, xmax = x_idx.min().float(), x_idx.max().float()
                ymin, ymax = y_idx.min().float(), y_idx.max().float()
                w_obj = (xmax - xmin + 1) / W
                h_obj = (ymax - ymin + 1) / H
                x_c = (xmin + xmax + 1) / (2 * W)
                y_c = (ymin + ymax + 1) / (2 * H)
                area_norm = w_obj * h_obj
                aspect = w_obj / (h_obj + 1e-6)
                geom_feats.append(torch.tensor([x_c, y_c, w_obj, h_obj, area_norm, aspect], device=device))
            geom_feats = torch.stack(geom_feats, dim=0)
            geom_feats_batch.append(geom_feats)
        return geom_feats_batch

    # =============================================================
    # Forward Pass
    # =============================================================
    def forward(self, x, obj_masks=None):
        # --- Extract visual features ---
        with torch.no_grad():
            res_feat = self.backbone(x)  # [B, 2048, H', W']

        # --- Pass through BSD for saliency refinement ---
        bsd_out = self.saliency_model(res_feat)
        if isinstance(bsd_out, dict) and 'saliency_map' in bsd_out:
            sal_feat = bsd_out['saliency_map']  # expected [B, 1, H', W']
        elif isinstance(bsd_out, dict) and 'logits_per_scale' in bsd_out:
            sal_feat = torch.sigmoid(bsd_out['logits_per_scale'][-1])  # take final scale
        else:
            sal_feat = bsd_out  # assume BSD outputs saliency directly

        sal_feat = F.interpolate(sal_feat, size=res_feat.shape[-2:], mode='bilinear', align_corners=False)
        salient_feats = res_feat * sal_feat  # saliency-weighted ResNet feature map

        # --- Object-level pooling ---
        B, C, H, W = salient_feats.shape
        obj_feats_batch = []
        if obj_masks is not None:
            geom_feats_batch = self.build_geometric_features(obj_masks, H, W, salient_feats.device)
            for i in range(B):
                feats = salient_feats[i]
                masks = obj_masks[i].to(feats.device)
                masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
                obj_feats = []
                for m in masks:
                    area = m.sum() + 1e-6
                    pooled = (feats * m).sum(dim=(1, 2)) / area
                    obj_feats.append(pooled)
                obj_feats = torch.stack(obj_feats, dim=0)
                # concatenate geometry before attention
                obj_feats = torch.cat([obj_feats, geom_feats_batch[i]], dim=-1)
                obj_feats_batch.append(obj_feats)
        else:
            obj_feats_batch = [salient_feats.mean(dim=(2, 3)).unsqueeze(1)]

        # --- Cross-attention between object embeddings ---
        rank_outputs = []
        for feats in obj_feats_batch:
            feats = self.pre_attn_proj(feats)
            feats = self.self_norm1(feats)

            if self.training:
                feats = feats + 0.01 * torch.randn_like(feats)
                feats = F.dropout(feats, p=0.1, training=True)

            if feats.shape[0] > 1:
                attended, _ = self.cross_attn(feats, feats, feats)
                feats = self.cross_norm(attended + feats)

            feats = self.self_norm2(feats)
            ranks = self.rank_head(feats).squeeze(-1)
            ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-6)  # normalize ranks
            rank_outputs.append(ranks)

        return {
            'rank_score': rank_outputs,
            'saliency_feature': sal_feat,
            'resnet_feature': res_feat
        }


class HierarchicalGNNRelModSRNet(nn.Module):
    def __init__(self, pretrained_saliency_model, feature_dim=256, attn_heads=4, dropout_p=0.2, threshold=0.5):
        super().__init__()
        self.saliency_model = pretrained_saliency_model
        for p in self.saliency_model.parameters():
            p.requires_grad = True

        self.aggregator = HierarchicalGraphAggregator(
            feature_dim=feature_dim,
            hidden_dim=feature_dim * 2,
            num_heads=attn_heads,
            dropout_p=dropout_p
        )

        self.threshold = threshold
        self.rank_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, 1)
        )

    def forward(self, x, bin_gt, obj_masks=None):
        sal_out = self.saliency_model(x, bin_gt)
        F_list = sal_out['F_per_scale']
        bin_loss = sal_out['loss']

        if obj_masks is None:
            raise ValueError("obj_masks required for object-level reasoning")

        # Hierarchical multi-scale reasoning
        obj_feats = self.aggregator(F_list, obj_masks)
        # obj_feats is a list: [tensor([N1, D]), tensor([N2, D]), ...]

        rank_outputs = [self.rank_head(f).squeeze(-1) for f in obj_feats]


        return {
            'rank_score': rank_outputs,
            'bin_loss': bin_loss
        }


# Example usage
# if __name__ == '__main__':
#     from bsd.bsd import MultiScaleSaliency
#     model_sal = MultiScaleSaliency(backbone_pretrained=True)
#     aggregator_model = AggregatedSaliencyRankNet(pretrained_saliency_model=model_sal)

#     x = torch.randn(2, 3, 512, 512)
#     out = aggregator_model(x)
#     print('Aggregated feature:', out['aggregated_feature'].shape)
#     print('Rank score:', out['rank_score'].shape)
#     print('Final saliency map:', out['logits_final'].shape)
