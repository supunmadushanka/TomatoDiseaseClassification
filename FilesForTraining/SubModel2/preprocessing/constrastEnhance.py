# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 11:59:02 2022

@author: ishar
"""
#  CLAHE

import cv2


def clahe_filter(img):
    # Convert RGB to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Split Channels
    l,a,b = cv2.split(lab)

    # Apply to l channel
    clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(8,8))
    lab_planes = clahe.apply(l)

    # Merge channels
    lab = cv2.merge((lab_planes,a,b))

    # Convert LAB to RGB
    cl_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    print ("Successfully used CLAHE algorithm")
    return cl_img
