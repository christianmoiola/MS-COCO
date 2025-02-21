import torch
from tqdm import tqdm

from config import DEVICE

def eval(model, data, lang):
    #model.eval()
    
    with torch.no_grad():
        with torch.autocast(DEVICE):
            pbar = tqdm(data)
            correctly_classified = 0
            total = 0
            for sample in pbar:   
                outputs = model(sample['input_ids'], sample['attention_mask'], sample['pixel_values'])
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                predicted_labels = probs.argmax(dim=1)  # the label with the highest probability is our prediction
                ground_truth = sample['category']
                # print(f"Probs: {probs}")
                # print(f"Predicted labels: {predicted_labels}, Ground truth: {ground_truth}")
                correctly_classified += (predicted_labels == ground_truth).sum().item()
                # print(f"Correctly classified: {correctly_classified}")
                total += predicted_labels.size(0)
                # print(f"Total: {total}")
                pbar.set_description(f"Accuracy: {correctly_classified/total}")



                