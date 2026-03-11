import torch

# def variable_collate_fn(batch):

#     img_ids = []
#     pil_images = []        # <-- now PILs, do NOT stack
#     gts_list = []
#     rank_lists = []
#     gt_class_list = []
#     inst_masks_list = []
#     rois_list = []
#     phrases_list = []

#     # ------------------------------------
#     # Unpack each sample in the batch
#     # ------------------------------------
#     for item in batch:
#         img_id, img_pil, gts, ranks, gt_classes, inst_masks, rois, phrases = item
        
#         img_ids.append(img_id)
#         pil_images.append(img_pil)         # <--- leave PIL images untouched
#         gts_list.append(gts)               # tensor
#         rank_lists.append(ranks)
#         gt_class_list.append(gt_classes)
#         inst_masks_list.append(inst_masks) # list of instance mask tensors
#         rois_list.append(rois)
#         phrases_list.append(phrases)

#     # ------------------------------------
#     # Stack ONLY tensors (not PIL images)
#     # ------------------------------------
#     gts_batch = torch.stack(gts_list, dim=0)  # (B,1,H,W)

#     # Do NOT stack inst_masks—they are variable-length per image
#     # Do NOT stack rois—they are variable-length
#     # Do NOT stack rank lists

#     return (
#         img_ids,
#         pil_images,        # <-- List of PIL images for HF Mask2Former
#         gts_batch,         # full GT masks (tensor)
#         rank_lists,
#         gt_class_list,
#         inst_masks_list,   # list of tensors
#         rois_list,
#         phrases_list
#     )

def variable_collate_fn(batch):

    img_ids = []
    pil_images = []        # list of PIL Images
    gts_list = []          # list of GT tensors (variable size)
    rank_lists = []
    gt_class_list = []
    inst_masks_list = []
    rois_list = []
    phrases_list = []

    for item in batch:
        img_id, img_pil, gts, ranks, gt_classes, inst_masks, rois, phrases = item

        img_ids.append(img_id)
        pil_images.append(img_pil)         # keep PIL
        gts_list.append(gts)               # DO NOT stack
        rank_lists.append(ranks)
        gt_class_list.append(gt_classes)
        inst_masks_list.append(inst_masks)
        rois_list.append(rois)
        phrases_list.append(phrases)

    return (
        img_ids,
        pil_images,        # List[PIL.Image]
        gts_list,          # List[Tensor(1,H,W)]
        rank_lists,
        gt_class_list,
        inst_masks_list,   # List[Tensor(N_i,1,H,W)]
        rois_list,
        phrases_list
    )

