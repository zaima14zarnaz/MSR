#from Desktop.research1.QAGNet.Dataset.IRSR_ASSR import description
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import os, json
import numpy as np
import random

import csv
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt", quiet=True)

class SaliencyDataset(Dataset):
    def __init__(self, image_dir, rank_dir, obj_seg_json, transform=None, img_size=(512, 512), roi_offset=15, descriptions_csv=None):
        self.image_dir = image_dir
        self.rank_dir = rank_dir
        self.img_size = img_size
        self.transform = transform
        self.roi_offset = roi_offset
        self.descriptions_csv = descriptions_csv
        # Collect image filenames
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Load segmentation list → dictionary {img_id: object_data}
        with open(obj_seg_json, 'r') as f:
            seg_list = json.load(f)
        self.obj_seg_dict = {item["img"]: item["object_data"] for item in seg_list}

        # Shared transforms
        self.img_transform = transform or transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        

        self.description_phrases = {}  # maps filename → list of trimmed phrases

        if self.descriptions_csv is not None:
            MAX_WORDS = 30  # safe upper bound for CLIP (≈ <77 tokens)
            with open(self.descriptions_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)

                for row in reader:
                    # CSV format: [image_id, image_filename, description]
                    image_filename = row[1]
                    img_description = row[2]

                    # Split into sentences
                    raw_phrases = sent_tokenize(img_description)

                    trimmed_phrases = []
                    for phrase in raw_phrases:
                        words = phrase.split()

                        if len(words) > MAX_WORDS:
                            # truncate at a word boundary
                            phrase = " ".join(words[:MAX_WORDS])
                        # phrase = self.remove_relational_words(phrase)

                        trimmed_phrases.append(phrase)

                    # store trimmed phrases
                    self.description_phrases[image_filename] = trimmed_phrases
            self.rng = random.Random(42)
   
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_id = os.path.splitext(img_name)[0]

        # --- Load image ---
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # --- Load objects (each has bbox, segmentation, category_id, etc.) ---
        obj_info_list = self.obj_seg_dict.get(img_id, [])

        # --- Load rank order ---
        rank_order = []
        rank_path = os.path.join(self.rank_dir, f"{img_id}.json")
        if os.path.exists(rank_path):
            with open(rank_path, "r") as f:
                rank_info = json.load(f)
                rank_order = rank_info.get("rank_order", [])

        rank_to_index = {obj_idx: (len(rank_order) - 1 - r)
                        for r, obj_idx in enumerate(rank_order)}

        # --- Masks, ROIs, and category IDs ---
        gt_bin_mask = Image.new("L", (w, h), 0)
        draw_bin = ImageDraw.Draw(gt_bin_mask)

        obj_masks_pil, rois_xyxy, gt_classes = [], [], []  # << added gt_classes list

        # insert roi filtering loop here. 

        for obj_idx, obj in enumerate(obj_info_list):
            segs = obj.get("segmentation", [])
            bbox = obj.get("bbox", None)
            category_id = obj.get("category_id", -1)
            if category_id == -1:
                category_id = 1

            if bbox is None:
                continue

            # --- Create blank mask ---
            obj_mask = Image.new("L", (w, h), 0)
            draw_obj = ImageDraw.Draw(obj_mask)

            has_valid_seg = False

            # --- Draw segmentation polygons ---
            if isinstance(segs, list):
                for seg in segs:
                    if len(seg) >= 6:
                        poly = [int(v) for v in seg]
                        draw_obj.polygon(poly, outline=1, fill=1)
                        has_valid_seg = True

                        # Also write to global bin mask (optional)
                        if obj_idx in rank_to_index:
                            draw_bin.polygon(poly, outline=1, fill=1)

            # --- If mask is EMPTY, do NOT add ROI or mask ---
            if not has_valid_seg:
                continue

            # --- If segmentation exists, add ROI + mask + class ---
            x, y, bw, bh = bbox
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            x1 = max(0, x1 - self.roi_offset)
            y1 = max(0, y1 - self.roi_offset)
            x2 = min(w, x2 + self.roi_offset)
            y2 = min(h, y2 + self.roi_offset)

            rois_xyxy.append([0, x1, y1, x2, y2])
            gt_classes.append(category_id)
            obj_masks_pil.append(obj_mask)

        # --- Transform image and masks ---
        image = self.img_transform(image)
        H_new, W_new = image.shape[-2], image.shape[-1]

        # --- Build single binary GT mask ---
        if len(obj_masks_pil) > 0:
            obj_arrays = [np.array(m, dtype=np.uint8) for m in obj_masks_pil]
            combined = np.maximum.reduce(obj_arrays)
            gt_bin_mask = Image.fromarray(combined)
        else:
            gt_bin_mask = Image.new("L", (w, h), 0)
        gt_bin_mask = self.mask_transform(gt_bin_mask).gt(0).float()

        if obj_masks_pil:
            obj_masks = []
            for m in obj_masks_pil:
                m = m.resize((W_new, H_new), resample=Image.NEAREST)
                m_t = transforms.functional.to_tensor(m)
                obj_masks.append(m_t)
            if obj_masks:
                obj_masks = torch.stack(obj_masks, dim=0)
            else:
                obj_masks = torch.zeros((0, 1, H_new, W_new), dtype=torch.float32)
        else:
            obj_masks = torch.zeros((0, 1, *self.img_size), dtype=torch.float32)

        # --- Rescale ROIs ---
        if len(rois_xyxy) > 0:
            sx, sy = W_new / float(w), H_new / float(h)
            rois = torch.tensor(rois_xyxy, dtype=torch.float32)
            rois[:, [1, 3]] *= sx
            rois[:, [2, 4]] *= sy
            rois[:, [1, 3]] = rois[:, [1, 3]].clamp(0, W_new)
            rois[:, [2, 4]] = rois[:, [2, 4]].clamp(0, H_new)
        else:
            rois = torch.zeros((0, 5), dtype=torch.float32)

        # --- Category tensor ---
        if len(gt_classes) > 0:
            gt_class_tensor = torch.tensor(gt_classes, dtype=torch.long)
            gt_class_tensor -= 1  # shift from [1–90] → [0–89]
        else:
            gt_class_tensor = torch.zeros((0,), dtype=torch.long)


        # --- Rank tensor ---
        gt_rank_tensor = torch.tensor(rank_order, dtype=torch.long)

        # --- Phrases ---
        # todo: Extract phrases from your csv file here
        phrases = self.description_phrases.get(img_name, [])  # get list of phrases for this image
        # phrases = ['panda','panda','panda','panda','panda','panda','panda','panda','panda','panda']
        # phrases = ["","","","","","","","","",""]
        # phrases =  []
        # phrases = ["66666666666666666666666666666666666666666666666666666666666666666666-6-6-6-6666 the666 the6 66666 66 666 of666666666666666666666666666666666666666666666666666666666666666666666666666666666"]
        # phrases = ["The image shows a cat having a good time while the dog is playing with the ball.", "Beside the river, there is a tall tree with beautiful green leaves and ripe orange vines."]
        # phrases = ["There is no one in the image"]
        # <== SHUFFLE PHRASES HERE
        # self.rng.shuffle(phrases) 
        return img_id, image, gt_bin_mask, gt_rank_tensor, gt_class_tensor, obj_masks, rois, phrases
