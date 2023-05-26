import cv2
import numpy as np

def yellow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_range = np.array([20, 100, 100])
    upper_range = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_range, upper_range)
    yellow_area = cv2.bitwise_and(img, img, mask = yellow_mask)
    print ("Successfully applied yellow mask")
    return yellow_area







# =============================================================================
# img = cv2.imread('dataset/archive/train/Leaf_Mold/image1.jpg', cv2.COLOR_BGR2RGB)
# preprocessed_img = preprocessed.preprocesed_image(img, 256)
# img = preprocessed_img
# cv2.imshow('image', img)
# 
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 
# lower_range = np.array([20, 100, 100])
# upper_range = np.array([30, 255, 255])
# yellow_mask = cv2.inRange(hsv, lower_range, upper_range)
# yellow_area = cv2.bitwise_and(img, img, mask = yellow_mask)
# cv2.imshow('mask', yellow_mask)
# cv2.imshow('yellow_area', yellow_area)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================
 
