# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 01:22:56 2022

@author: ishar
"""

import cv2
from FilesForTraining.SubModel2.preprocessing.resize import resize_image
from FilesForTraining.SubModel2.preprocessing.bgRemove import bg_remove
from FilesForTraining.SubModel2.preprocessing.bilateralFilter import bilateral_filter
from FilesForTraining.SubModel2.preprocessing.constrastEnhance import clahe_filter


def preprocesed_image(img, SIZE):
   
    resize_img = resize_image(img, SIZE)    

    bgRemove_img = bg_remove(resize_img)
   
    bilateral_img = bilateral_filter(bgRemove_img)
    
    contrast_img = clahe_filter(bilateral_img)
    
    print ("Successfully preprocessed image")
    return contrast_img


