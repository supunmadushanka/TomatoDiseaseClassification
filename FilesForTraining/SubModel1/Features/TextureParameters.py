# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 22:21:53 2023

@author: Supun Madushanka
"""

import cv2
import numpy as np


def TextureParameters(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate the mean and standard deviation of the Laplacian
    mean, std_dev = cv2.meanStdDev(laplacian)
    
    # Compute variance of Laplacian to estimate lesion texture
    variance = np.var(laplacian)
    
    # Print the feature values
    print("Laplacian mean:", mean[0][0])
    print("Laplacian standard deviation:", std_dev[0][0])
    
    return ([mean[0][0],std_dev[0][0], variance])
