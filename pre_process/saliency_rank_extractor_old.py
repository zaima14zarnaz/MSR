"""
Docstring for sr-model1.dataset_builder.py.saliency_rank_extractor
Build saliency ranking dataset from saliency maps
"""
import torch
import os
import numpy as np
import cv2
from PIL import Image
import json
from scipy.stats import mode
import cv2
from pycocotools import mask as maskUtils
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
splits = {"test":"test", "train":"train", "val":"val"}
dataset_dir = "/data/research/zaima/dataset/Dataset/SIFR/SIFR_dataset"
# dataset_dir = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR"
# dataset_dir = "/data/research/zaima/dataset/Dataset/ASSR"
image_store = "images"
sal_map_store = "gt"
ranks_order_store = "rank_order"

coco_ann_file = "/data/research/zaima/dataset/Dataset/coco_annotations/instances_val2017.json"
with open(coco_ann_file, "r") as f:
    coco = json.load(f)

def segmentation_to_mask(segmentation, height, width):
    """
    Convert any COCO segmentation (polygon, RLE, uncompressed RLE)
    into a binary mask of shape (H, W).
    """

    # Case 1: POLYGON segmentation
    if isinstance(segmentation, list):
        # it may be a list of polygon lists
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return m.astype(np.uint8)

    # Case 2: RLE segmentation (compressed or uncompressed)
    elif isinstance(segmentation, dict) and "counts" in segmentation:
        # If counts is a list → UNCOMPRESSED RLE → convert
        if isinstance(segmentation["counts"], list):
            rle = maskUtils.frPyObjects(segmentation, height, width)
        else:
            # counts is a string → already compressed
            rle = segmentation

        m = maskUtils.decode(rle)
        return m.astype(np.uint8)

    else:
        raise ValueError(f"Unknown segmentation type: {type(segmentation)}")
    
def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union

from pycocotools import mask as coco_mask
def ann_to_mask(ann, H, W):
    rle = coco_mask.frPyObjects(ann["segmentation"], H, W)
    mask = coco_mask.decode(rle)

    if mask.ndim == 3:
        mask = np.any(mask, axis=2)

    return mask.astype(bool)



def extract_masks_from_sal_map(image_id, image_dir):
    img = np.array(Image.open(image_dir).convert("L"))

    unique_vals = [v for v in np.unique(img) if v > 0]

    objects = {image_id: []}
    next_id = 1

    for val in unique_vals:
        mask = (img == val).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue

        mask_cv = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coco_polys = []
        all_points = []

        for cnt in contours:
            if len(cnt) >= 3:
                poly = cnt.reshape(-1, 2).astype(float)
                coco_polys.append(poly.flatten().tolist())
                all_points.append(poly)

        if not coco_polys:
            continue

        all_points = np.vstack(all_points)
        x_min = float(all_points[:, 0].min())
        y_min = float(all_points[:, 1].min())
        x_max = float(all_points[:, 0].max())
        y_max = float(all_points[:, 1].max())

        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        obj = {
            "segmentation": coco_polys,
            "area": area,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": bbox,
            "category_id": 0,
            "id": next_id
        }

        objects[image_id].append(obj)
        next_id += 1

    return objects


    
def extract_saliency_rank_order(image_dir, objects):
    """
    Returns:
        rank_order: indices within the valid-mask list, sorted by descending saliency
        valid_masks: original mask indices that were retained (in original order)
        invalid_masks: original mask indices that were removed (in original order)
    """
    print(image_dir)
    img = np.array(Image.open(image_dir).convert("L"))
    H, W = img.shape

    valid_entries = []   # (original_idx, mode_value)
    invalid_masks = []   # original indices only

    for idx, object in enumerate(objects):
        seg = object['segmentation'] 
        # print(seg)
        full_mask = segmentation_to_mask(seg, H, W)
        vals = img[full_mask.astype(bool)]

        if len(vals) == 0:
            invalid_masks.append(idx)
            continue

        mode_val = mode(vals, keepdims=False).mode

        if mode_val > 0:
            valid_entries.append((idx, int(mode_val)))
        else:
            invalid_masks.append(idx)

    # valid mask original indices (in original order)
    valid_masks = [x[0] for x in valid_entries]

    # Sort valid entries by saliency DESCENDING
    sorted_valid = sorted(
        enumerate(valid_entries),   # enumerate so we keep valid-list index
        key=lambda x: x[1][1],      # sort by mode_val
        reverse=True
    )

    # rank_order: indices within valid list
    rank_order = [i for (i, entry) in sorted_valid]
    print(rank_order)

    return rank_order, valid_masks, invalid_masks



def extract_objects(image_dir, coco):
    image_name = os.path.splitext(os.path.basename(image_dir))[0]
    image_id = None

    for img in coco.get("images", []):
        if os.path.splitext(img["file_name"])[0] == image_name:
            image_id = img["id"]
            break

    # if image_id is None:
    #     return []

    seg_info = extract_masks_from_sal_map(image_id=image_id, image_dir=image_dir)

    if image_id not in seg_info or len(seg_info[image_id]) == 0:
        return []

    H, W = Image.open(image_dir).size[::-1]

    objects = []

    anns = [ann for ann in coco.get("annotations", []) if ann["image_id"] == image_id]

    # --------------------------------------------------
    # CASE 1: NO annotations → directly use seg_info
    # --------------------------------------------------
    if len(anns) == 0:
        for idx, seg_entry in enumerate(seg_info[image_id]):
            obj = {
                "segmentation": seg_entry["segmentation"],
                "area": seg_entry["area"],
                "bbox": seg_entry["bbox"],
                "iscrowd": 0,
                "image_id": image_id,
                "category_id": -1,
                "id": seg_entry.get("id", idx),
                "matched_iou": None
            }
            objects.append(obj)

        return objects

    # --------------------------------------------------
    # CASE 2: annotations available → IoU matching
    # --------------------------------------------------
    for ann in anns:
        ann_mask = ann_to_mask(ann, H, W)

        best_iou = 0.0
        best_idx = -1

        for idx, seg_entry in enumerate(seg_info[image_id]):
            seg_mask = segmentation_to_mask(
                seg_entry["segmentation"], H, W
            ).astype(bool)

            iou = mask_iou(seg_mask, ann_mask)

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou < 0.8 or best_idx < 0:
            continue

        matched = seg_info[image_id][best_idx]

        obj = {
            "segmentation": matched["segmentation"],
            "area": matched["area"],
            "bbox": matched["bbox"],
            "iscrowd": ann.get("iscrowd", 0),
            "image_id": image_id,
            "category_id": ann.get("category_id", -1),
            "id": ann.get("id", -1),
            "matched_iou": best_iou
        }

        objects.append(obj)

    return objects


def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

split = splits["test"]
images_dir = os.path.join(dataset_dir, image_store, split)
image_files = os.listdir(images_dir)

save_test_anns = os.path.join(dataset_dir, f"obj_seg_data_{split}.json")
save_rank_order_dir = os.path.join(dataset_dir, ranks_order_store, split)

test_anns = []
for img_file in tqdm(image_files, desc="Processing images"):
    basename = os.path.splitext(img_file)[0]
    sal_map_dir = os.path.join(dataset_dir, sal_map_store, split, f"{basename}.png")
    if not os.path.exists(sal_map_dir):
        continue
    # print(sal_map_dir)
    objects = extract_objects(sal_map_dir, coco=coco)
    
    saliency_ranks, valid_objects, invalid_objects = extract_saliency_rank_order(sal_map_dir, objects)
    anns = {
        "img": basename,
        "object_data": objects
    }
    test_anns.append(anns)
    rank_order_info = {
        "image_id": basename, 
        "rank_order": saliency_ranks
    }
    rank_order_fname = os.path.join(save_rank_order_dir, f"{basename}.json")
    save_json(rank_order_info, rank_order_fname)
    # break
    
    
    # print(f"Filename: {basename}, saliency_ranks: {saliency_ranks} of {len(valid_objects)} objects")

save_json(test_anns, save_test_anns)



    




