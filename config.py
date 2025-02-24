import os

# PATHS
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_MODELS = os.path.join(ROOT_DIR, "models")
PATH_IMAGES = os.path.join(ROOT_DIR, "coco/images/val2017")
PATH_ANNOTATIONS = os.path.join(ROOT_DIR, "coco/annotations/instances_val2017.json")
# END PATHS

# CONFIG
DEVICE = "mps"
CLIP = "openai/clip-vit-large-patch14-336"
# END CONFIG

# SETTINGS
LABEL = "category" # POSSIBLE VALUES: "category" or "supercategory"
TEMPLATE = "The element in the image surrounded by a rectangle is a {}" # "A photo of a {}"
EXTRACTION_BBOX = "draw_rectangle" # POSSIBLE VALUES: "crop" or "draw_rectangle"
THRESHOLD_BBOX = 300 # POSSIBLE VALUES: int or None
# END SETTINGS