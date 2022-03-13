from cv2_utils import resize_img, inflate_boxes, preprocess, read_img, resize_gray_img
from vision_utils import get_bounds, get_vision_response, plot_bounds, FeatureType
from object_detection import detect_objects, plot_boxes, intersecting_area, area
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from pathlib import Path
import cv2
from copy import deepcopy

def imshow(img, cmap = None):
    plt.figure(dpi = 250)
    plt.imshow(img, cmap = cmap)
    plt.axis('off')
    plt.show()
    
img_files = [img_path for img_path in Path('/Users/hariharan/Downloads/12.02.2022').rglob('*.jpg')]
len(img_files)

# 1. 'IMG_20220126_103739.jpg'
# 2. IMG_20220126_103843.jpg
# image_file = '/Users/hariharan/hari_works/grocery-object-detection/ocr/madhan_images/' 
image_file = '/Users/hariharan/Downloads/12.02.2022/IMG_20220212_172559_1.jpg'
img_rgb = read_img(image_file)
img_for_detection = resize_img(img_rgb, new_height = 640) # 4608

s = time.time()
selected_boxes, selected_labels = detect_objects(img_for_detection, print_preds_times = False)
# inflate selected boxes
selected_boxes = inflate_boxes(selected_boxes, img_rgb.shape[:2], 640)
print(selected_boxes)
e = time.time()
print('bounding box times: ', (e-s), ' seconds')

copy_img = deepcopy(img_rgb)
boxed_img = plot_boxes(copy_img, selected_boxes, selected_labels)
Image.fromarray(boxed_img)

mask = np.zeros_like(img_rgb)
for (ymin, xmin, ymax, xmax) in selected_boxes:
    cv2.rectangle(mask, (xmin, ymax), (xmax, ymin), (1, 1, 1), -1)
test = mask * img_rgb
gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# gray = resize_gray_img(gray, new_height = 2048)
thresh = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
Image.fromarray(thresh)

s = time.time()
document = get_vision_response(thresh) # s
e = time.time()
print('time for prediction: ', e-s, ' seconds')

bounds, texts = get_bounds(document, FeatureType.BLOCK)
bounds = inflate_boxes(np.array(bounds), img_rgb.shape[:2], new_height = 2048)
# (boxes * factor).astype(int)
copy_img = deepcopy(img_rgb)
copyOut = plot_bounds(copy_img, bounds)
Image.fromarray(copyOut)


box_groupings = {','.join([str(i) for i in box]): [] for box in selected_boxes.tolist()}
box_groupings

for text_block, text in zip(bounds, texts):
    for b in box_groupings:
        box_coords = [int(i) for i in b.split(',')]
        inter = intersecting_area(text_block, box_coords)
        if inter is not None:
            inter_percent = inter/ area(text_block)
            if inter_percent > 0.95:
                box_groupings[b].append(text)
                
box_groupings

