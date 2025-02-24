import torch
from tqdm import tqdm
from sklearn.metrics import classification_report


from config import *

def eval(model, data, lang):
    all_preds = []
    all_labels = []
    pbar = tqdm(data)
    pbar.set_description("Evaluation")

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

        all_preds.extend(predicted_labels.cpu().tolist())
        all_labels.extend(ground_truth.cpu().tolist())

    if LABEL == "category":
        target_names = lang.category
    elif LABEL == "supercategory":
        target_names = lang.superCategory
    
    return classification_report(all_labels, all_preds, target_names=target_names)