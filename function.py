import torch
from torcheval.metrics import MulticlassF1Score
from tqdm import tqdm

from config import DEVICE

def eval(model, data, lang):
    #model.eval()
    
    with torch.no_grad():
        print("Len ", len(lang.category))

        f1_metric = MulticlassF1Score(num_classes=len(lang.category), average='macro').to(DEVICE)

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

                # Accuracy
                correctly_classified += (predicted_labels == ground_truth).sum().item()
                total += predicted_labels.size(0)

                # F1 Score
                f1_metric.update(predicted_labels, ground_truth)
                #current_f1 = f1_metric.compute().item()

                pbar.set_description(f"Acc:{correctly_classified/total}")



                