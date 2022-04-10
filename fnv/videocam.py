from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
# url = 'http://192.168.1.9:8080/video'

url = '/Users/hariharan/Downloads/test_demo.mov'
cap = cv2.VideoCapture(url)

# Object detection from Stable Camera.
object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 40)

while(True):
    ret, frame = cap.read()

    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate areas and remove small elements.
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('mask',  mask)

    key = cv2.waitKey(30)
    print(key)
    if key == 27:
        break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
cap.release()

cv2.destroyAllWindows()