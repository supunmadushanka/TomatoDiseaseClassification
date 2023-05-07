# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:35:26 2023

@author: Supun Madushanka
"""

import cv2
import numpy as np


def GradientFeature(img):

    # Load the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute the gradient using the Sobel operator
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute the magnitude and direction of the gradient
    mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    
    # Compute the mean and standard deviation of the gradient magnitude
    mean_mag = np.mean(mag)
    std_mag = np.std(mag)
    
    # Compute the mean and standard deviation of the gradient direction
    mean_angle = np.mean(angle)
    std_angle = np.std(angle)
    
    # Return the results
    return ([mean_mag, std_mag, mean_angle, std_angle])
