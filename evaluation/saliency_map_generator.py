import json
import os

import os
import json
import numpy as np
import cv2

sal_map_dir = "/data/research/zaima/dataset/Dataset/IRSR_dataset/dataset/saliency_maps"
os.makedirs(sal_map_dir, exist_ok=True)

saved_data_path = "sr_ranks.json"
with open(saved_data_path, "r") as f:
    gt_data = json.load(f)

height, width = 512, 512

def polygons_to_mask(polygons, h, w):
    """Convert COCO polygons -> binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask


# ----------------------------------------------------------
# MAIN: Generate saliency maps per image
# ----------------------------------------------------------
for image_id, image_data in gt_data.items():

    objects = image_data["object_data"]
    sal_map = np.zeros((height, width), dtype=np.uint8)

    ranks = [obj["pred_rank"] for obj in objects if obj["pred_rank"] is not None]
    if len(ranks) == 0:
        continue

    min_rank = min(ranks)
    max_rank = max(ranks)
    range_rank = max_rank + 1 - min_rank if max_rank != min_rank else 1
    print(image_id)
    for obj in objects:
        polys = obj.get("pred_segmentation", None)
        rank = obj.get("pred_rank", None)

        if polys is None or rank is None:
            continue

        mask = polygons_to_mask(polys, height, width)

        # invert rank: high-saliency (low rank) → brighter
        inv = max_rank - rank + 1
        gray_val_reverse = int((inv / range_rank) * 254) + 1  # in [1,255]
        print(" rank", rank, "-> gray", gray_val_reverse)

        sal_map[mask == 1] = gray_val_reverse

    # if image_id == "COCO_val2014_000000000192":
    #     print("----", image_id, "----")
    #     for obj in objects:
    #         r = obj["pred_rank"]
    #         if obj.get("pred_segmentation") is None:
    #             print(" rank", r, "SKIPPED (no segmentation)")
    #             continue
    #         inv = max_rank - r + 1
    #         gray = int((inv / range_rank) * 254) + 1
    #         print(" rank", r, "-> gray", gray)


    out_path = os.path.join(sal_map_dir, f"{image_id}.png")
    cv2.imwrite(out_path, sal_map)
    # print(f"Saved saliency map: {out_path}")

        