import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
import numpy as np
import os, json, csv
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

class SaliencyDataset(Dataset):
    def __init__(
        self,
        image_dir,
        rank_dir,
        obj_seg_json,
        transform=None,
        img_size=(512, 512),
        roi_offset=15,
        descriptions_csv=None,
    ):
        self.image_dir = image_dir
        self.rank_dir = rank_dir
        self.img_size = img_size
        self.roi_offset = roi_offset
        self.descriptions_csv = descriptions_csv

        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Load COCO-style segmentation json
        with open(obj_seg_json, 'r') as f:
            seg_list = json.load(f)
        self.obj_seg_dict = {item["img"]: item["object_data"] for item in seg_list}

        # ❗ Mask2Former must NOT use this transform → keep only mask resize
        self.img_transform = None  

        # self.resize_mask = transforms.Resize(img_size, interpolation=Image.NEAREST)

        # Optional text descriptions
        self.description_phrases = {}
        if descriptions_csv is not None:
            with open(descriptions_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    image_filename = row[1]
                    desc = row[2]
                    self.description_phrases[image_filename] = sent_tokenize(desc)

        # Build mapping category_id → label_idx
        all_ids = []
        for seg in seg_list:
            for obj in seg["object_data"]:
                all_ids.append(obj["category_id"])

        unique_ids = sorted(set(all_ids))
        self.cat2idx = {cid: i for i, cid in enumerate(unique_ids)}

    def __len__(self):
        return len(self.images)

    def _make_binary_mask(self, pil_mask):
        arr = np.array(pil_mask, dtype=np.uint8)
        return Image.fromarray((arr > 0).astype(np.uint8) * 255)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_id = os.path.splitext(img_name)[0]

        # ---------------------------
        # Load ORIGINAL IMAGE (PIL)
        # ---------------------------
        img_path = os.path.join(self.image_dir, img_name)
        image_pil = Image.open(img_path).convert("RGB")
        w, h = image_pil.size

        # ---------------------------
        # Load coco annotation
        # ---------------------------
        obj_info_list = self.obj_seg_dict.get(img_id, [])
        
        # Load rank info
        rank_order = []
        rank_json = os.path.join(self.rank_dir, f"{img_id}.json")
        if os.path.exists(rank_json):
            with open(rank_json, "r") as f:
                rank_order = json.load(f).get("rank_order", [])

        obj_masks_list = []
        rois = []
        gt_classes = []

        full_gt_mask = Image.new("L", (w, h), 0)
        draw_gt = ImageDraw.Draw(full_gt_mask)

        # ---------------------------
        # Build masks + ROIs
        # ---------------------------
        for obj_idx, obj in enumerate(obj_info_list):

            bbox = obj.get("bbox", None)
            segs = obj.get("segmentation", [])
            category_id = obj.get("category_id", -1)
            if bbox is None:
                continue

            obj_mask = Image.new("L", (w, h), 0)
            draw_obj = ImageDraw.Draw(obj_mask)

            has_poly = False
            if isinstance(segs, list):
                for seg in segs:
                    if len(seg) >= 6:
                        poly = [int(v) for v in seg]
                        draw_obj.polygon(poly, fill=1)
                        draw_gt.polygon(poly, fill=1)
                        has_poly = True

            if not has_poly:
                continue

            # ROI
            x, y, bw, bh = bbox
            x1, y1 = max(0, x - self.roi_offset), max(0, y - self.roi_offset)
            x2, y2 = min(w, x + bw + self.roi_offset), min(h, y + bh + self.roi_offset)
            rois.append([0, x1, y1, x2, y2])

            gt_classes.append(self.cat2idx[category_id])
            obj_masks_list.append(obj_mask)

        # ---------------------------
        # Resize masks for MAE evaluation
        # ---------------------------
        if len(obj_masks_list) > 0:
            inst_masks = []
            for m in obj_masks_list:
                # m = self.resize_mask(m)
                m = self._make_binary_mask(m)
                inst_masks.append(TF.to_tensor(m))
            inst_masks = torch.stack(inst_masks, dim=0)  # (N,1,H,W)
        else:
            inst_masks = torch.zeros((0,1,self.img_size[0],self.img_size[1]))

        # Full-image GT mask
        # full_gt_mask = self.resize_mask(full_gt_mask)
        full_gt_mask = self._make_binary_mask(full_gt_mask)
        gts = TF.to_tensor(full_gt_mask).float()

        # Resize ROIs to mask grid
        if len(rois) > 0:
            rois = torch.tensor(rois, dtype=torch.float32)
            sx, sy = self.img_size[0] / h, self.img_size[1] / w
            rois[:, [1, 3]] *= sx
            rois[:, [2, 4]] *= sy
        else:
            rois = torch.zeros((0,5))

        # Classes
        gt_class_tensor = torch.tensor(gt_classes, dtype=torch.long) \
                          if len(gt_classes) > 0 else torch.zeros((0,), dtype=torch.long)

        phrases = self.description_phrases.get(img_name, [])

        # ---------------------------
        # RETURN ORIGINAL + RESIZED
        # ---------------------------
        return (
            img_id,
            image_pil,     # <── ORIGINAL IMAGE (PIL!)
            gts,           # resized full mask
            torch.tensor(rank_order, dtype=torch.long),
            gt_class_tensor,
            inst_masks,
            rois,
            phrases,
        )
