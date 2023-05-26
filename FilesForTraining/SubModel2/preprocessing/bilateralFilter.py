# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:57:32 2022

@author: ishar
"""

import cv2


def bilateral_filter(img):
    bilateralFilter_image = cv2.bilateralFilter(img,9,25,25)
    print ("Successfully used bilateral filter")
    return bilateralFilter_image
