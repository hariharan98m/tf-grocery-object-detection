import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import tensorflow_hub as hub
import pandas as pd
import numpy as np

# Loading model directly from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

def area(rect):
    ymin, xmin, ymax, xmax = rect
    dx = xmax - xmin
    dy = ymax - ymin
    return dx * dy

def intersecting_area(a, b):  # returns None if rectangles don't intersect
    ymin_1, xmin_1, ymax_1, xmax_1 = a
    ymin_2, xmin_2, ymax_2, xmax_2 = b
    dx = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
    dy = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
    if (dx>=0) and (dy>=0):
        return dx*dy

def plot_boxes(img_rgb, boxes, labels, print_preds_times = False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()
    # Putting the boxes and labels on the image
    for (ymin, xmin, ymax, xmax), label in zip(boxes, labels):
        img_boxes = cv2.rectangle(img_rgb,(xmin, ymax),(xmax, ymin),(0,255,0), 2)
        # cv2.putText(img_boxes, label, (xmin, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
    end = time.time()
    if print_preds_times:
        print(f'total plotting time: {round(end-start, 2)}s')

    return img_rgb

def remove_boxes(selected_boxes):
    remove = []
    for a_index, a in enumerate(selected_boxes):
        inner_box = []
        for b_index, b in enumerate(selected_boxes):
            if a_index != b_index:
                inter = intersecting_area(a, b)
                a_area = area(b)
                if inter is not None and float(inter)/a_area > 0.8:
                    inner_box.append(b_index)
        if len(inner_box) > 1:
            remove.append(a_index)
        if len(inner_box) == 1:
            remove.append(inner_box[0])
    return remove

# Loading csv with labels of classes
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

def detect_objects(img_rgb, print_preds_times = False):
    # Converting to uint8
    rgb_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)

    # Creating prediction
    start = time.time()
    boxes, scores, classes, num_detections = detector(rgb_tensor)
    end = time.time()
    if print_preds_times:
        print(f'total preds time: {round(end-start, 2)}s')

    selected_indices = tf.image.non_max_suppression(
        boxes = boxes[0], scores = scores[0], max_output_size = 100, iou_threshold=0.5,
        score_threshold=0.1, name=None
    )
    selected_boxes = tf.gather(boxes[0], selected_indices).numpy().astype(int)
    selected_scores = tf.gather(scores[0], selected_indices).numpy()
    selected_classes = tf.gather(classes[0], selected_indices).numpy()

    start = time.time()
    removes = remove_boxes(selected_boxes)
    end = time.time()
    if print_preds_times:
        print(f'total removes time: {round(end-start, 2)}s')

    for i, cls in enumerate(selected_classes):
        if labels[int(cls)] == 'dining table':
            removes.append(i)

    selected_boxes = np.delete(selected_boxes, removes, axis = 0)
    selected_scores = np.delete(selected_scores, removes, axis = 0)
    selected_classes = np.delete(selected_classes, removes, axis = 0)

    selected_labels = [labels[int(i)] for i in selected_classes]

    return selected_boxes, selected_labels