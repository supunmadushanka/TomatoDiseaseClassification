# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 21:38:43 2023

@author: Supun Madushanka
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def TextureParametersSobel(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Compute gradients in x and y directions
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients to obtain edges
    edges = cv2.addWeighted(np.abs(grad_x), 0.5, np.abs(grad_y), 0.5, 0)
    # plt.imshow(edges)
    
    # Compute variance of edges to estimate lesion texture
    variance = np.var(edges)
    
    # Calculate the mean and standard deviation of the Laplacian
    mean, std_dev = cv2.meanStdDev(edges)
    
    return ([mean[0][0],std_dev[0][0], variance])
