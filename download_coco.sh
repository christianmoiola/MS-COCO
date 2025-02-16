# val data
wget http://images.cocodataset.org/zips/val2017.zip -P coco/images
unzip coco/images/val2017.zip -d coco/images

# annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P coco/annotations
unzip coco/annotations/annotations_trainval2017.zip -d coco