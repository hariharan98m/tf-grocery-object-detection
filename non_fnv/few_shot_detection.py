import tensorflow as tf
import numpy as np
import cv2
import time
from cv2_utils import resize_img, inflate_boxes, preprocess, read_img, resize_gray_img

from fsdet.engine import DefaultPredictor
from fsdet.config import get_cfg
cfg = get_cfg()
config_file = 'configs/Grocery-detection/faster_rcnn_R101_FPN_ft_all.yaml'
cfg.merge_from_file(config_file)
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

labels = [ # all classes
        "3roses_top_star",
        "dettol_250ml",
        "dettol_500ml",
        "guntur_chilli",
        "hamam_100g",
        "maha_jyothi",
        "moong_dhal",
        "nescafe_classic",
        "nescafe_sunrise",
        "super_shine",
        "vim_soap"
    ]

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

def plot_boxes(img_rgb, boxes, labels, print_preds_times = False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()
    # Putting the boxes and labels on the image
    for (ymin, xmin, ymax, xmax), label in zip(boxes, labels):
        cv2.rectangle(img_rgb,(xmin, ymax),(xmax, ymin),(0,255,0), 2)
        cv2.putText(img_rgb, label, (xmin, ymin), font,
                    3, (0, 0, 255), 5, cv2.LINE_AA)
    end = time.time()
    if print_preds_times:
        print(f'total plotting time: {round(end-start, 2)}s')

    return img_rgb

def detect_objects(img):
    # predict bounding box.
    pred = predictor(img)

    classes = tf.convert_to_tensor(pred['instances'].pred_classes.cpu().numpy().astype(int), dtype= tf.float32)
    scores = tf.convert_to_tensor(pred['instances'].scores.cpu().numpy().astype(float), dtype = tf.float32)

    boxes = pred['instances'].pred_boxes.tensor.cpu().numpy()
    boxes = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]]).transpose(1, 0)
    boxes = tf.convert_to_tensor(boxes, dtype = tf.float32)

    selected_indices = tf.image.non_max_suppression(
            boxes = boxes, scores = scores, max_output_size = 100, iou_threshold=0.5,
            score_threshold=0.1, name=None
        )

    selected_boxes = tf.gather(boxes, selected_indices).numpy()
    selected_scores = tf.gather(scores, selected_indices).numpy()
    selected_classes = tf.gather(classes, selected_indices).numpy()

    removes = remove_boxes(selected_boxes)

    selected_boxes = np.delete(selected_boxes, removes, axis = 0)
    selected_scores = np.delete(selected_scores, removes, axis = 0)
    selected_classes = np.delete(selected_classes, removes, axis = 0)

    selected_labels = [labels[int(i)] for i in selected_classes]

    return selected_boxes, selected_scores, selected_labels

if __name__ == '__main__':
    image_file = f'/Users/hariharan/hari_works/grocery-object-detection/non_fnv/some_test_images/IMG_20220214_161632.jpg'
    img_rgb = read_img(image_file)
    img_for_detection = resize_img(img_rgb, new_height = 640) # 4608

    s = time.time()
    selected_boxes, selected_scores, selected_labels = detect_objects(img_for_detection)
    e = time.time()
    print('object detect times:', (e-s)/10.)
    overhead = 0.2

    selected_boxes = inflate_boxes(selected_boxes, img_rgb.shape[:2], 640)
    img_rgb = plot_boxes(img_rgb, selected_boxes, [f'Object_{p}' for p in range(len(selected_boxes))] )

    # show img.
    cv2.imshow("Result", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)