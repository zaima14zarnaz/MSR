import torch
def variable_collate_fn(batch):
    img_ids, imgs, gts, rank_lists, class_tensors, masks, rois, phrases = zip(*batch)

    imgs = torch.stack(imgs, dim=0)
    gts = torch.stack(gts, dim=0)

    rank_lists = list(rank_lists)

    rois_with_batch = []
    gt_classes_all = []

    for b_idx, (r, cls_tensor) in enumerate(zip(rois, class_tensors)):
        if r.numel() > 0:
            r = r.clone()
            r[:, 0] = b_idx
            rois_with_batch.append(r)
            gt_classes_all.append(cls_tensor.clone())
        else:
            rois_with_batch.append(torch.zeros((0, 5), dtype=torch.float32))
            gt_classes_all.append(torch.zeros((0,), dtype=torch.long))

    img_mask = gts

    # ✨ NEW: keep phrases exactly as-is, but convert tuple → list
    phrases = list(phrases)

    return (
        img_ids,          # same
        imgs,             # same
        gts,              # same
        rank_lists,       # same
        gt_classes_all,   # same
        img_mask,         # same
        masks,            # same
        rois_with_batch,  # same
        phrases           # ✨ now returned properly
    )
