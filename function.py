import torch
from tqdm import tqdm
from torcheval.metrics import MulticlassF1Score


from config import *

def eval(model, data, lang):
    #model.eval()
    f1_metric = MulticlassF1Score(num_classes=len(lang.category), average='macro').to(DEVICE)

    
    pbar = tqdm(data)
    correctly_classified = 0
    total = 0
    for sample in pbar:
        with torch.no_grad():
            with torch.autocast(DEVICE):   
                outputs = model(sample['input_ids'], sample['attention_mask'], sample['pixel_values'])
        
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        predicted_labels = probs.argmax(dim=1)  # the label with the highest probability is our prediction

        if LABEL == "category":
            ground_truth = sample['category']
        elif LABEL == "supercategory":
            ground_truth = sample['supercategory']

        # Accuracy
        correctly_classified += (predicted_labels == ground_truth).sum().item()
        total += predicted_labels.size(0)

        # F1 Score
        f1_metric.update(predicted_labels, ground_truth)

        pbar.set_description(f"Acc:{correctly_classified/total}")

    txt = f"Model:{CLIP} Label:{LABEL}_Template:{TEMPLATE}_Extraction_bbox:{EXTRACTION_BBOX}_Threshold:{THRESHOLD_BBOX}_Acc:{correctly_classified/total}_F1:{f1_metric.compute()}"
    with open("results.txt", "a") as f:
        f.write(txt + "\n")

    print(f"Analysis performance of {CLIP} on the MS COCO dataset with {LABEL} as label")
    print(f"The template used is:{TEMPLATE}")
    print(f"The method of extraction of the bounding box is {EXTRACTION_BBOX}")
    if THRESHOLD_BBOX:
        print(f"The threshold of the bounding box is {THRESHOLD_BBOX}")
    print(f"Accuracy: {correctly_classified/total}, F1 Score: {f1_metric.compute()}")