import os
import xml.etree.ElementTree as ET

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fsdet.utils.file_io import PathManager

__all__ = ["register_meta_pascal_voc"]

from pathlib import Path

def grocery_dataset_loader(name, thing_classes):
    data = []

    base_dir = 'grocery-db'
    # base_dir = '/home/ubuntu/tf-grocery-object-detection/non_fnv/grocery-db' # on the AWS GPU machine.

    # thing_classes = ["3roses_top_star", "dettol_250ml", "hamam_100g", "nescafe_classic"]
    # thing_classes = ["3roses_top_star", "dettol_250ml", "hamam_100g", "nescafe_classic"]

    index = 0

    for cls in thing_classes:
        for img_file in (Path(base_dir) / cls).rglob('*.jpg'):
            imgname = img_file.stem
            anno_file = str(Path(base_dir) / cls / (imgname + '.xml'))

            tree = ET.parse(anno_file)
            size = tree.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)

            annts = []
            for object in tree.findall('object'):
                class_name = object.find('name').text
                boundbox = object.find('bndbox')
                xmin = float(boundbox.find('xmin').text)
                xmax = float(boundbox.find('xmax').text)
                ymin = float(boundbox.find('ymin').text)
                ymax = float(boundbox.find('ymax').text)
                annts.append({
                    "category_id" : thing_classes.index(class_name), # class unique ID
                    "bbox" : [xmin, ymin, xmax, ymax], # bbox coordinates
                    "bbox_mode" : BoxMode.XYXY_ABS, # bbox mode, depending on your format
                })
            data_instance = {
                "file_name" : str(img_file), # full path to image
                "image_id" :  index, # image unique ID
                "height" : height, # height of image
                "width" : width, # width of image
                "annotations": annts
            }
            data.append(data_instance)
            index += 1
    return data
