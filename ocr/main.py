import cv2
import numpy as np

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

import pytesseract
import pdb
# img = cv2.imread('/Users/hariharan/Downloads/sign_text.png')

def map_to_product(text):
    t2p = {
        'milk rusk': ['milk', 'rusk', 'elite', 'good', 'for', 'you'],
        'honey': ['dabur', 'extra', 'worl', 'extra'],
        'moisturizer': ['nivea', 'men', 'dark', 'spot', 'moisturiser'],
        'refresh tears': ['refresh', 'tears', 'carboxy', 'methyl'],
        'credit card': ['millennia', 'hdfc', 'hariharan', 'manikandan'],
        'ketchup': ['chai', 'kings', 'tomato'],
        'oinment': ['cutisoft', 'hydrocortisone', 'cream', 'ip']
    }
    for t, keywords in t2p.items():
        for k in keywords:
            if k in text.lower():
                return t
    return None

# Setup capture
# cap = cv2.VideoCapture(0)
# cap.set(3,640) #1280
# cap.set(4,480) #480

# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         continue
from pathlib import Path
for p in Path('testimages').glob('*.jpg'):
    img = cv2.imread(str(p))
    imgname = p.stem
    # img = np.array(frame)

    h, w, c = img.shape
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    text = pytesseract.image_to_string(img)
    product = map_to_product(text)
    print(text, product)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, product, (10,450), font, 3, (0, 255, 0), 10, cv2.LINE_AA)

    # cv2.imshow('Output',cv2.resize(img, (640, 480)))
    cv2.imwrite('result/' + imgname + '.png', cv2.resize(img, (640, 480)))
        # cv2.waitKey(1)