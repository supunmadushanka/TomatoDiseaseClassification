# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 21:36:28 2023

@author: ishar
"""

from rembg import remove
import cv2

def bg_remove(img):
    img = remove(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) 
    print ("Successfully removed background")
    return img
