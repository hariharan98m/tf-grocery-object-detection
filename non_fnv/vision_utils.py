import argparse
from enum import Enum
import io
import numpy as np
import cv2
from google.cloud import vision
from PIL import Image

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

# client_options = {'api_endpoint': 'eu-vision.googleapis.com'}
# import pdb
# pdb.set_trace()
# from google.oauth2 import service_account
# credentials = service_account.Credentials.from_service_account_file('/Users/hariharan/hari_works/grocery-object-detection/ocr/retailzz-4d18b61af484.json')
# client = vision.ImageAnnotatorClient(credentials=credentials)

# client = vision.ImageAnnotatorClient() #client_options = client_options
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('/Users/hariharan/Downloads/grocery_keys/retailzz-4d18b61af484.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

def get_vision_response(img):
    vision_image = vision.Image(content=cv2.imencode('.jpg', img)[1].tostring())
    response = client.document_text_detection(image= vision_image)
    document = response.full_text_annotation
    # del client
    return document

def bound2box(bound):
    xs = [bound.vertices[i].x for i in range(4)]
    ys = [bound.vertices[i].y for i in range(4)]
    # ymin, xmin, ymax, xmax
    return [min(ys), min(xs), max(ys), max(xs)]

# Collect specified feature bounds by enumerating all document features
def get_bounds(document, feature):
    bounds = []; texts = []
    for page in document.pages:
        page_text = ''
        for block in page.blocks:
            block_text = ''
            for paragraph in block.paragraphs:
                paragraph_text = ''
                for word in paragraph.words:
                    word_text = ''
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            bounds.append(bound2box(symbol.bounding_box))
                        word_text+= symbol.text

                    if feature == FeatureType.WORD:
                        bounds.append(bound2box(word.bounding_box))
                        texts.append(word_text)
                    paragraph_text+= (word_text + ' ')

                if feature == FeatureType.PARA:
                    bounds.append(bound2box(paragraph.bounding_box))
                    texts.append(paragraph_text)
                block_text+= (paragraph_text + '\n')

            if feature == FeatureType.BLOCK:
                bounds.append(bound2box(block.bounding_box))
                texts.append(block_text)
            page_text+= (block_text + '\n\n')

        if feature == FeatureType.PAGE:
            bounds.append(bound2box(block.bounding_box))

    return bounds, texts

def plot_bounds(img_rgb, bounds):
    # Blue color in BGR
    color = (255, 0, 0)
    thickness = 2
    for bound in bounds:
        ymin, xmin, ymax, xmax = bound
        # xs = [bound.vertices[i].x for i in range(4)]
        # ys = [bound.vertices[i].y for i in range(4)]
        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img_rgb = cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
    return img_rgb
