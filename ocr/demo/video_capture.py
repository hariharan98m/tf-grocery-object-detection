import argparse
import datetime
from object_detection import detect_objects, plot_boxes
from cv2_utils import resize_img, inflate_boxes
import time
import cv2
import pdb
# url = 'http://192.168.1.9:8080/video'
object_detector = cv2.createBackgroundSubtractorMOG2() #history = 100, varThreshold = 40

url = '/Users/hariharan/Downloads/test_demo.mov'
cap = cv2.VideoCapture(url)

frame_counter = 0

from tracker import *

tracker = EuclideanDistTracker()

while True:

    # 1. Object Detection
    detections = []
    ret, frame = cap.read()
    frame_resized = resize_img(frame)
    # pdb.set_trace()
    boxes, labels = detect_objects(frame_resized)
    resized_boxes = inflate_boxes(boxes, frame.shape[:2])
    for (ymin, xmin, ymax, xmax) in boxes:
        detections.append([xmin, ymin, (xmax - xmin), (ymax - ymin)])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    print(boxes_ids)

    # mask = object_detector.apply(frame)
    # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # cv2.imshow('mask',  mask)

    plot_boxes(frame, resized_boxes, labels, print_preds_times= False)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    frame_counter+=1
cap.release()
cv2.destroyAllWindows()