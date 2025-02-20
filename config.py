import os

# PATHS
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_MODELS = os.path.join(ROOT_DIR, "models")
PATH_IMAGES = os.path.join(ROOT_DIR, "coco/images/val2017")
PATH_ANNOTATIONS = os.path.join(ROOT_DIR, "coco/annotations/instances_val2017.json")

# CONFIG
DEVICE = "mps"
CLIP = "openai/clip-vit-large-patch14-336"
TEMPLATE = "A photo of a {}"



