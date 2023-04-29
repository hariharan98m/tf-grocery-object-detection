import cv2
from PIL import Image
import numpy as np

def resize_img(img_rgb, new_height = 2048.):
    length_x, width_y, _ = img_rgb.shape
    factor = min(1, float(new_height / length_x))
    size = int(factor * width_y), int(factor * length_x)
    # resize image
    img_rgb_resized = cv2.resize(img_rgb, size, Image.ANTIALIAS)
    # cv2.imshow('after plotting boxes', img_rgb_resized)
    # cv2.waitKey()
    return img_rgb_resized

def inflate_boxes(boxes, original_size, new_height):
    height, width = original_size
    # new_height = 2048. #120.
    factor = height / new_height
    return (boxes * factor).astype(int)

def read_img(image_file, color = True):
    #Load image by Opencv2
    img = cv2.imread(image_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # length_x, width_y, _ = img_rgb.shape
    # factor = min(1, float(2048.0 / length_x))
    # size = int(factor * width_y), int(factor * length_x)
    # resize image
    # img_rgb_resized = cv2.resize(img_rgb, size, Image.ANTIALIAS)
    return img_rgb #_resized

def resize_gray_img(img_gray, new_height = 2048):
    length_x, width_y = img_gray.shape
    factor = min(1, float(new_height / length_x))
    size = int(factor * width_y), int(factor * length_x)
    # resize image
    img_rgb_resized = cv2.resize(img_gray, size, Image.ANTIALIAS)
    return img_rgb_resized

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # img = image_smoothening(img)
    img = cv2.medianBlur(img,5)
    or_image = cv2.bitwise_or(img, filtered) #closing
    return or_image