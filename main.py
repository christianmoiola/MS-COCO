from transformers import CLIPProcessor


from utils import *

device = "mps" 


if __name__ == "__main__":
    dataType='val2017'
    annFile='coco/annotations/instances_{}.json'.format(dataType)
    coco=load_coco(annFile)

    raw_dataset = create_coco_dataset(coco=coco, path_images="coco/images/val2017")

    category = set([d['category'] for d in raw_dataset])
    superCategory = set([d['supercategory'] for d in raw_dataset])
    
    lang = Lang(category, superCategory)

    print(raw_dataset[0])

    path = os.path.join(os.getcwd(), "models")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=path)

    dataset = Ms_Coco(raw_dataset, lang, processor, template="A photo of a {}")

    print(dataset.__getitem__(0)["input_ids"].shape)
    # print(dataset.__len__())