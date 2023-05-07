# -*- coding: utf-8 -*-
"""
Created on Mon May  1 12:27:03 2023

@author: Supun Madushanka
"""

import cv2
from skimage.feature import hog
import pandas as pd

def HogFilter(img):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate HOG feature
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    
    df = pd.DataFrame(fd)

    df1 = df.transpose()
    
    return ([df1, hog_image])