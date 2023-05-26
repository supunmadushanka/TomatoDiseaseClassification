# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:58:10 2023

@author: ishar
"""

import cv2

def resize_image(img, SIZE):
    img = cv2.resize(img, (SIZE, SIZE)) 
    print ("Successfully resized image")
    return img


