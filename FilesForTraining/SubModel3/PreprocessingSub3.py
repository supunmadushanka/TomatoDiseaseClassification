# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 01:22:56 2022

@author: ishar
"""

import cv2
from rembg import remove 

#PRE-PROCESSOR function
def pre_processor(input_img):
    

    def medianFilter(img):
       denoised = cv2.medianBlur(img, 3)
       return denoised


    def equalizer(img):
        lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        #Splitting the LAB image to L, A and B channels, respectively
        l, a, b = cv2.split(lab_img)

        ###########CLAHE#########################
        #Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(l)
        #plt.hist(clahe_img.flat, bins=100, range=(0,255))

        #Combine the CLAHE enhanced L-channel back with A and B channels
        updated_lab_img2 = cv2.merge((clahe_img,a,b))

        #Convert LAB image back to color (RGB)
        CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
        return CLAHE_img
        
    blur = medianFilter(input_img)
    CLAHE_img = equalizer(blur)
    b_removed = remove(CLAHE_img)
    output= cv2.cvtColor(b_removed , cv2.COLOR_BGR2RGB)
    return output


