import torch
from tqdm import tqdm

from config import DEVICE

def eval(model, data, lang):
    #model.eval()
    
    with torch.no_grad():
        with torch.autocast(DEVICE):
            pbar = tqdm(data)
            for sample in pbar:
                #print(sample["input_ids"].shape, sample["attention_mask"].shape, sample["pixel_values"].shape)    
                outputs = model(sample['input_ids'], sample['attention_mask'], sample['pixel_values'])
                #logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                #probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                #print(probs.shape)