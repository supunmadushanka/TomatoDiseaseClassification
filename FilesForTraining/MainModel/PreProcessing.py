# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 20:07:05 2022

@author: Supun Madushanka
"""

#import numpy as np
import cv2
from matplotlib import pyplot as plt
from rembg import remove
from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.filters import sobel
import numpy as np



def bg_remove(img):
    new_img = img_as_ubyte(img)
    bg_rem_img = remove(new_img)
    bg_rem_img_BGR = cv2.cvtColor(bg_rem_img, cv2.COLOR_BGRA2BGR)
    float_img = img_as_float(bg_rem_img_BGR)
    return float_img

def bg_remove_int(img):
    bg_rem_img = remove(img)
    bg_rem_img_BGR = cv2.cvtColor(bg_rem_img, cv2.COLOR_BGRA2BGR)
    return bg_rem_img_BGR



def segment_disease(img):
    new_img = img_as_ubyte(img)
    hsv_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36,0,0])
    upper_green = np.array([170,255,255])
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    mask1 = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(new_img,new_img, mask= mask1)
    float_img = img_as_float(result)
    return float_img

def segment_disease_int(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36,0,0])
    upper_green = np.array([170,255,255])
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    mask1 = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img,img, mask= mask1)
    return result

def segment_disease_with_range(img, lb, ub):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([lb,0,0])
    upper_green = np.array([ub,255,255])
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    mask1 = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img,img, mask= mask1)
    return result
    
def image_smooth(img):
    new_img = img_as_ubyte(img)
    gaussian_using_cv2 = cv2.GaussianBlur(new_img, (3,3), 0, borderType=cv2.BORDER_CONSTANT)
    float_img = img_as_float(gaussian_using_cv2)
    return float_img

def image_smooth_int(img):
    gaussian_using_cv2 = cv2.GaussianBlur(img, (3,3), 0, borderType=cv2.BORDER_CONSTANT)
    return gaussian_using_cv2



def sobel_filter(img):
    edge_sobel = sobel(img)
    return img_as_ubyte(edge_sobel)



def improve_contrast(img):
    new_img = img_as_ubyte(img)
    
    lab= cv2.cvtColor(new_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    float_img = img_as_float(enhanced_img)

    return float_img

def improve_contrast_int(img):
    
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img



def draw_hist(img):
    plt.hist(img.flat, bins=256, range=(0,255))
    

def eqa_hist(gray_img):
    equ = cv2.equalizeHist(gray_img)
    return equ