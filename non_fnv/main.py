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
from nlp import filter_stopwords, match_vals, product_list, find_product

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

if __name__ == '__main__':
    root = Tk()
    root.withdraw()
    image_file = fd.askopenfilename()
    root.update()
    root.destroy()

    # image_file = '/Users/hariharan/hari_works/grocery-object-detection/ocr/demo/14.02.2022/IMG_20220214_162302.jpg' #'14.02.2022/IMG_20220214_162228.jpg'
    img_rgb = read_img(image_file)
    img_for_detection = resize_img(img_rgb, new_height = 640) # 4608

    s = time.time()
    selected_boxes, selected_scores, selected_labels = detect_objects(img_for_detection)
    e = time.time()
    print('object detect times:', (e-s)/10.)
    overhead = 0.2

    s = time.time()
    selected_boxes = inflate_boxes(selected_boxes, img_rgb.shape[:2], 640)

    # objects - with threshold, gray, color.
    color_objs = []
    for (ymin, xmin, ymax, xmax) in selected_boxes:
        color_objs.append(img_rgb[ymin:ymax, xmin:xmax])

    with ThreadPool(20) as p:
        documents = list(tqdm.tqdm(p.imap(ocr_mp, color_objs), total = len(color_objs)))

    selected_boxes, final_products, variants = find_product(selected_boxes, documents)
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
        bounds[:, 0]+= ymin; bounds[:, 2]+= ymin; bounds[:, 1]+= xmin; bounds[:, 3]+= xmin
        img_rgb = plot_bounds(img_rgb, bounds)

    img_rgb = plot_boxes(img_rgb, selected_boxes, [p if p is not None else '' for p in final_products] )
    e = time.time()
    print()
    print('Total Time taken: ', round(e-s + overhead, 3), ' sec\n')
    print('Total Objects detected: ', int(num_objects_detected))

    # show img.
    cv2.imshow("Result", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)