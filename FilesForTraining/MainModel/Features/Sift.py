# -*- coding: utf-8 -*-
"""
Created on Tue May  2 22:25:40 2023

@author: Supun Madushanka
"""

import cv2
import pandas as pd

# Load the image
def SiftFilter(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize the SIFT feature extractor
    sift = cv2.SIFT_create(nfeatures=200)
    
    # Detect and compute SIFT features
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    new_descriptors=descriptors.ravel()
    
    # Display the SIFT keypoints
    # img_sift = cv2.drawKeypoints(gray, keypoints, img)
    
    df = pd.DataFrame(new_descriptors)

    df1 = df.transpose()
    
    return df1