import os
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.utils.data as data
from pycocotools.coco import COCO
from torch.utils.data import default_collate

from config import *

def load_coco():
    return COCO(PATH_ANNOTATIONS)

def create_coco_dataset(coco):
    imgs = coco.loadImgs(coco.getImgIds())

    dict_idcat_supercat = {}
    dict_id_cat = {}
    cats = coco.loadCats(coco.getCatIds())
    for cat in cats:
        dict_idcat_supercat[cat['id']] = cat['supercategory']
        dict_id_cat[cat['id']] = cat['name']

    dataset = []

    for img in tqdm(imgs, desc="Extraction dataset"):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"]))
        original_image = Image.open(os.path.join(PATH_IMAGES, img["file_name"]))
        for ann in anns:
            idcat = ann['category_id']
            x, y, w, h = ann['bbox']

            if THRESHOLD_BBOX is not None:
                if w < THRESHOLD_BBOX or h < THRESHOLD_BBOX:
                    continue

            if EXTRACTION_BBOX =="crop":
                processed_image = original_image.crop((x, y, x + w, y + h))
            elif EXTRACTION_BBOX == "draw_rectangle":
                processed_image = original_image.copy()
                draw = ImageDraw.Draw(processed_image)
                draw.rectangle([x, y, x + w, y + h], outline="red")
            else:
                raise ValueError(f"Extraction method not supported: {EXTRACTION_BBOX}")

            dataset.append({
                'image': processed_image,
                'category': dict_id_cat[idcat],
                'supercategory': dict_idcat_supercat[idcat]
            })
    return dataset

class Lang():
    def __init__(self, category, superCategory):
        self.category2id = self.lab2id(category)
        self.superCategory2id = self.lab2id(superCategory)
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.id2superCategory = {v: k for k, v in self.superCategory2id.items()}
        self.category = category
        self.superCategory = superCategory

    def lab2id(self, labels):
        vocab={}
        for label in labels:
            vocab[label] = len(vocab)
        return vocab

class Ms_Coco(data.Dataset):
    def __init__(self, dataset, lang, processor):
        self.lang = lang
        self.processor = processor

        self.images = []
        self.categories = []
        self.supercategories = []

        for el in dataset:
            self.images.append(el['image'])
            self.categories.append(lang.category2id[el['category']])
            self.supercategories.append(lang.superCategory2id[el['supercategory']])
        
        if LABEL == "category":
            self.text = [TEMPLATE.format(cat) for cat in lang.category]
        elif LABEL == "supercategory":
            self.text = [TEMPLATE.format(cat) for cat in lang.superCategory]
        else:
            raise ValueError(f"Label not supported: {LABEL}")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        category = self.categories[idx]
        supercategory = self.supercategories[idx]
        image = self.images[idx]
        processed_image = self.processor(text=self.text, images=image, return_tensors="pt", padding=True)

        return {"input_ids": processed_image["input_ids"].to(DEVICE),
                "attention_mask": processed_image["attention_mask"].to(DEVICE),
                "pixel_values": processed_image["pixel_values"][0].to(DEVICE),
                "category": torch.tensor(category).to(DEVICE),
                "supercategory": torch.tensor(supercategory).to(DEVICE)}
    

def collate_fn(data):
    new_item = {}
    input_ids = data[0]["input_ids"]
    attention_mask = data[0]["attention_mask"]
    data = default_collate(data)

    pixel_values = data["pixel_values"]
    category = data["category"]
    supercategory = data["supercategory"]

    new_item["input_ids"] = input_ids
    new_item["attention_mask"] = attention_mask
    new_item["pixel_values"] = pixel_values
    new_item["category"] = category
    new_item["supercategory"] = supercategory
    return new_item


def save_classification_report(report):
    if not os.path.exists(os.path.join(ROOT_DIR, "results")):
        os.makedirs(os.path.join(ROOT_DIR, "results"))
    if not os.path.exists(os.path.join(ROOT_DIR, f"results/{LABEL}")):
        os.makedirs(os.path.join(ROOT_DIR, f"results/{LABEL}"))
    with open(os.path.join(ROOT_DIR, f"results/{LABEL}/classification_report_{EXTRACTION_BBOX}_{TEMPLATE}_{THRESHOLD_BBOX}.txt"), "w") as f:
        f.write(f"CONFIG\n")
        f.write(f"DEVICE: {DEVICE}\n")
        f.write(f"CLIP: {CLIP}\n")
        f.write(f"LABEL: {LABEL}\n")
        f.write(f"TEMPLATE: {TEMPLATE}\n")
        f.write(f"EXTRACTION_BBOX: {EXTRACTION_BBOX}\n")
        f.write(f"THRESHOLD_BBOX: {THRESHOLD_BBOX}\n")
        f.write(f"\n")
        f.write(f"CLASSIFICATION REPORT\n")
        f.write(report)
        f.write(f"\n")
        f.write(f"END CLASSIFICATION REPORT\n")