# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:51:01 2023

@author: Supun Madushanka
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def Brown_Area_And_Circularity(img):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define the color range for the lesion areas
    lower = np.array([0, 50, 50])
    upper = np.array([20, 255, 255])
    
    # Create a binary mask for the lesion areas
    mask = cv2.inRange(hsv, lower, upper)
    
    # Apply morphological operations to remove noise and smooth the edges of the lesion areas
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # plt.imshow(mask)
    
    # Calculate the perimeter and circularity of the lesion areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    perimeters = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        perimeters.append(perimeter)
        
    indices = np.argsort(perimeters)[::-1][:5]
        
    if len(contours) > 0:
        contour1 = contours[indices[0]]
        perimeter1 = perimeters[indices[0]]
        area1 = cv2.contourArea(contour1)
        circularity1 = (4 * 3.1416 * area1) / (perimeter1 ** 2)
    else:
        area1=0
        perimeter1 = 0
        circularity1 = -1
        
    if len(contours) > 1 :
        contour2 = contours[indices[1]]
        perimeter2 = perimeters[indices[1]]
        area2 = cv2.contourArea(contour2)
        circularity2 = (4 * 3.1416 * area2) / (perimeter2 ** 2)
    else:
        area2=0
        perimeter2 = 0
        circularity2 = -1
        
    if len(contours) > 2 :
        contour3 = contours[indices[2]]
        perimeter3 = perimeters[indices[2]]
        area3 = cv2.contourArea(contour3)
        circularity3 = (4 * 3.1416 * area3) / (perimeter3 ** 2)
    else:
        area3=0
        perimeter3 = 0
        circularity3 = -1
        
    if len(contours) > 3 :
        contour4 = contours[indices[3]]
        perimeter4 = perimeters[indices[3]]
        area4 = cv2.contourArea(contour4)
        circularity4 = (4 * 3.1416 * area4) / (perimeter4 ** 2)
    else:
        area4=0
        perimeter4 = 0
        circularity4 = -1

    if len(contours) > 4 :
        contour5 = contours[indices[4]]
        perimeter5 = perimeters[indices[4]]
        area5 = cv2.contourArea(contour5)
        circularity5 = (4 * 3.1416 * area5) / (perimeter5 ** 2)
    else:
        area5=0
        perimeter5 = 0
        circularity5 = -1
    
    return ([
        area1,
        circularity1,
        area2,
        circularity2,
        area3,
        circularity3,
        area4,
        circularity4,
        area5,
        circularity5,
        len(contours)
        ])
