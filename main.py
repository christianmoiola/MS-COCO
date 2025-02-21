from transformers import CLIPProcessor
from torch.utils.data import DataLoader

from config import *
from utils import *
from function import *
from model import ClipModel



if __name__ == "__main__":
    torch.manual_seed(1234)

    coco=load_coco()

    raw_dataset = create_coco_dataset(coco=coco)
    #raw_dataset = raw_dataset[3870:3890]

    category = set([d['category'] for d in raw_dataset])
    superCategory = set([d['supercategory'] for d in raw_dataset])
    
    lang = Lang(category, superCategory)

    processor = CLIPProcessor.from_pretrained(CLIP, cache_dir=PATH_MODELS)
    dataset = Ms_Coco(raw_dataset, lang, processor)
    
    dataloader = DataLoader(dataset, batch_size=2,  shuffle=True, collate_fn=collate_fn)

    clip = ClipModel()

    eval(clip, dataloader, lang)

    # print(dataset.__len__())