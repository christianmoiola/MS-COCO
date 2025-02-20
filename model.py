import torch.nn as nn
from transformers import CLIPModel

from config import PATH_MODELS, DEVICE

class ClipModel(nn.Module):
    def __init__(self):
        super(ClipModel, self).__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=PATH_MODELS, device_map=DEVICE)

    def forward(self, input_ids, attention_mask, pixel_values):
        return self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)