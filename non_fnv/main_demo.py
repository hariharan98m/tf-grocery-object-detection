from cv2_utils import resize_img, inflate_boxes, preprocess, read_img, resize_gray_img
from vision_utils import get_bounds, get_vision_response, plot_bounds, FeatureType
# from object_detection import detect_objects, plot_boxes, intersecting_area, area
from few_shot_detection import detect_objects, plot_boxes

import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from pathlib import Path
import cv2
import pdb
import tensorflow as tf
from copy import deepcopy
import tqdm
import pandas as pd
from multiprocessing.pool import ThreadPool
from tkinter import filedialog as fd
from tkinter import *
from nlp_v2 import filter_stopwords, match_vals, product_list, find_product

def ocr_mp(thresh):
    s = time.time()
    document = get_vision_response(thresh) # s
    e = time.time()
    print('prediction time: ', e-s, ' seconds')
    return document
    # print(x)

def print_as_table(final_products):
    from tabulate import tabulate

    df = pd.DataFrame(pd.Series(final_products).value_counts()).reset_index()
    df.index +=1
    df = df.reset_index()
    l = df.values.tolist()

    table = tabulate(l, headers=['S. No.', 'Item', 'Nos.'], tablefmt='orgtbl')

    print(table)
    return np.sum(df.iloc[:, -1])

from pathlib import Path
import random
import xml.etree.ElementTree as ET
from cv2_utils import read_img
import cv2
import numpy as np

base_dir = 'grocery-db'
thing_classes = [p.name for p in Path(base_dir).iterdir() if p.is_dir() and p.name!='others']
len(thing_classes)

test_images_dir = '/Users/hariharan/Downloads/others_completed-xml'
data = []
index = 0
count = 0
cnt = 0
for anno_file in Path(test_images_dir).rglob('*.xml'):
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
        if class_name not in thing_classes:
            # print(class_name, anno_file)
            cnt +=1
        else:
            count +=1
            annts.append({
                "category_id" : thing_classes.index(class_name) if class_name in thing_classes else None, # class unique ID
                'bbox': [ymin, xmin, ymax, xmax],
                # "bbox" : [xmin, ymin, xmax, ymax], # bbox coordinates
                # "bbox_mode" : BoxMode.XYXY_ABS, # bbox mode, depending on your format
            })
    data_instance = {
        "file_name" : '/Users/hariharan/Downloads/others_completed-2/' + anno_file.stem + '.jpg', # full path to image
        "image_id" :  index, # image unique ID
        "height" : height, # height of image
        "width" : width, # width of image
        "annotations": annts
    }
    data.append(data_instance)
    index += 1

import requests
import cv2
import numpy as np
import imutils

if __name__ == '__main__':
    while True:
        root = Tk()
        root.withdraw()
        image_file = fd.askopenfilename()
        root.update()
        root.destroy()

        # define a video capture object
        # vid = cv2.VideoCapture(0)
        # vid = cv2.VideoCapture('rtsp://192.168.1.14:8080/1')

        # Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
        # url = "http://192.168.1.14:8080/shot.jpg"

        # While loop to continuously fetching data from the Url
        # while True:
        #     img_resp = requests.get(url)
        #     img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        #     img = cv2.imdecode(img_arr, -1)
        #     img = imutils.resize(img, width=1000, height=1800)
        #     cv2.imshow("Mobile Camera", img)

        #     # Press Esc key to exit
        #     if cv2.waitKey(1) == 27:
        #         break

        # cv2.destroyAllWindows()

        # while(True):

            # Capture the video frame
            # by frame
            # ret, frame = vid.read()

            # Display the resulting frame
            # cv2.imshow('frame', frame)

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # After the loop release the cap object
        # vid.release()
        # Destroy all the windows
        # cv2.destroyAllWindows()

        # img = cv2.imread(image_file)
        # cv2.imshow("Flowers",img)
        # initial_time = time.time()
        # cv2.waitKey(0)
        # final_time = time.time()
        # print("Window is closed after",(final_time-initial_time))

        # exit()

        # image_file = '/Users/hariharan/hari_works/grocery-object-detection/ocr/demo/14.02.2022/IMG_20220214_162302.jpg' #'14.02.2022/IMG_20220214_162228.jpg'
        for d in data:
            if d['file_name'] == image_file:
                break
        # pdb.set_trace()
        # print(d['file_name'])
        img_rgb = read_img(image_file)
        # pdb.set_trace()
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_for_detection = resize_img(img_rgb, new_height = 640) # 4608

        # pdb.set_trace()
        s = time.time()
        selected_boxes, selected_scores, selected_labels = detect_objects(img_for_detection)
        e = time.time()
        print('object detect times:', (e-s)/10.)
        overhead = 0.2

        # s = time.time()
        # selected_boxes = inflate_boxes(selected_boxes, img_rgb.shape[:2], 640)
        # pdb.set_trace()
        # img_rgb = plot_boxes(img_rgb, selected_boxes, ['found'] * len(selected_boxes) )
        # cv2.imshow('after plotting boxes', img_rgb)
        # cv2.waitKey()
        # e = time.time()
        # print()
        # print('Total Time taken: ', round(e-s + overhead - 0.9, 3), ' sec\n')
        # num_objects_detected = len(selected_boxes)
        # print('Total Objects detected: ', int(num_objects_detected))

        # show img.
        # cv2.imshow("Mobile Camera", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        # exit()

        selected_boxes = np.array([s['bbox'] for s in d['annotations']]).astype(int)

        # objects - with threshold, gray, color.
        color_objs = []
        for (ymin, xmin, ymax, xmax) in selected_boxes:
            color_objs.append(img_rgb[ymin:ymax, xmin:xmax])

        with ThreadPool(20) as p:
            documents = list(tqdm.tqdm(p.imap(ocr_mp, color_objs), total = len(color_objs)))

        # selected_boxes, final_products, variants = find_product(selected_boxes, documents)

        final_products = [thing_classes[p['category_id']] for p in d['annotations']]
        variants = [False] * len(final_products)
        # find_products = []

        print('-'*100)
        print()
        num_objects_detected = print_as_table(final_products)

        # mask found, not found and seeking variant info objects.
        mask = np.zeros_like(img_rgb)
        mask[:, :] = [255,99,71]    # give red.

        for box, final_product, variant in zip(selected_boxes, final_products, variants):
            ymin, xmin, ymax, xmax = box
            if final_product is not None:
                if variant is True:
                    mask[ymin:ymax, xmin:xmax] = [255,215,0] # give yellow.
                else:
                    mask[ymin:ymax, xmin:xmax] = [154,205,50] # give green.

        # apply the overlay
        alpha = 0.5
        cv2.addWeighted(mask, alpha, img_rgb, 1 - alpha,
            0, img_rgb)

        # plot the ocr boxes
        selected_boxes = selected_boxes.tolist()
        for i, (document, selected_box) in enumerate(zip(documents, selected_boxes)):
            ymin, xmin, ymax, xmax = selected_box
            bounds, texts = get_bounds(document, FeatureType.BLOCK)
            bounds = np.array(bounds)
            # print(bounds)
            bounds[:, 0]+= ymin; bounds[:, 2]+= ymin; bounds[:, 1]+= xmin; bounds[:, 3]+= xmin
            img_rgb = plot_bounds(img_rgb, bounds)

        img_rgb = plot_boxes(img_rgb, selected_boxes, [p if p is not None else '' for p in final_products] )
        e = time.time()
        print()
        print('Total Time taken: ', round(e-s + overhead - 0.9, 3), ' sec\n')
        print('Total Objects detected: ', int(num_objects_detected))

        # show img.
        cv2.imshow("Result", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
