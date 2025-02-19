import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import os
from PIL import Image
from tqdm import tqdm


def load_coco(path):
    '''
        input: path/to/data
        output: coco instance
    '''
    return COCO(path)

def create_coco_dataset(coco, path_images):
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
        original_image = Image.open(os.path.join(path_images, img["file_name"]))
        for ann in anns:
            idcat = ann['category_id']
            bbox = ann['bbox']
            image_cropped = original_image.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
            dataset.append({
                'image': image_cropped,
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
    def __init__(self, dataset, lang, processor, template="A photo of a {}"):
        self.lang = lang
        self.processor = processor

        self.images = []
        self.categories = []
        self.supercategories = []

        for el in dataset:
            self.images.append(el['image'])
            self.categories.append(lang.category2id[el['category']])
            self.supercategories.append(lang.superCategory2id[el['supercategory']])
        
        self.text = [template.format(cat) for cat in lang.category]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        category = self.categories[idx]
        supercategory = self.supercategories[idx]
        image = self.images[idx]
        processed_image = self.processor(text=self.text, images=image, return_tensors="pt", padding=True)

        return {"input_ids": processed_image["input_ids"],
                "attention_mask": processed_image["attention_mask"],
                "pixel_values": processed_image["pixel_values"],
                "category": torch.tensor(category),
                "supercategory": torch.tensor(supercategory)}