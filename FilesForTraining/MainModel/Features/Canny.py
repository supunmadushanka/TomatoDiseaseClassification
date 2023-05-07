# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:26:24 2023

@author: Supun Madushanka
"""

import cv2
import pandas as pd
import numpy as np


def CannyFeatures(img):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection with thresholds of 100 and 200
    edges = cv2.Canny(gray, 150, 320)
    
    # Calculate the mean, standard deviation, skewness and kurtosis of the edge image
    mean, std_dev = cv2.meanStdDev(edges)
    skewness = np.mean((edges - mean)**3) / std_dev**3
    kurtosis = np.mean((edges - mean)**4) / std_dev**4
    
    # Add the statistical features to a pandas DataFrame
    df = pd.DataFrame({
        'mean_edge_intensity': mean[0],
        'std_dev_edge_intensity': std_dev[0],
        'skewness_edge_intensity': skewness[0],
        'kurtosis_edge_intensity': kurtosis[0]
    }, index=[0])

    return([mean[0], std_dev[0], skewness[0], kurtosis[0]])
