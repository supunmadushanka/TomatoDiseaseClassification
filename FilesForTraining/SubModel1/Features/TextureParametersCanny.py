# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 21:37:14 2023

@author: Supun Madushanka
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def TextureParametersCanny(img):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 100, 200)
    plt.imshow(edges)
    
    # Compute variance of edges to estimate lesion texture
    variance = np.var(edges)
    
    # Calculate the mean and standard deviation of the Laplacian
    mean, std_dev = cv2.meanStdDev(edges)
    
    # Print the lesion texture feature
    print('Lesion texture:', variance)
    
    return ([mean[0][0],std_dev[0][0], variance])
