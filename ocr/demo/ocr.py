from cv2_utils import resize_img, inflate_boxes, preprocess, read_img
from vision_utils import get_bounds, get_vision_response, plot_bounds, FeatureType
from object_detection import detect_objects, plot_boxes, intersecting_area, area
import cv2

image_file = '/Users/hariharan/hari_works/grocery-object-detection/ocr/madhan_images/IMG_20220126_103739.jpg'
img_rgb = read_img(image_file)

selected_boxes, selected_labels = detect_objects(img_rgb, print_preds_times = False)
print(selected_boxes)

plot_boxes(img_rgb, selected_boxes, selected_labels)

cv2.imshow('image', img_rgb)
cv2.waitKey(0)

#closing all open windows
cv2.destroyAllWindows()

# document = get_vision_response()
# get_bounds(document, feature)