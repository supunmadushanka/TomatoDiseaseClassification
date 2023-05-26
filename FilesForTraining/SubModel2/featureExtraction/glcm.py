# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 17:51:23 2022

@author: ishar
"""
import numpy as np 
import cv2
import skimage.feature as feature


def glcm(image_spot):
    gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)

    # Find the GLCM
    graycom = feature.greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    # Find the GLCM properties
    contrast = feature.greycoprops(graycom, 'contrast')
    dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
    homogeneity = feature.greycoprops(graycom, 'homogeneity')
    energy = feature.greycoprops(graycom, 'energy')
    correlation = feature.greycoprops(graycom, 'correlation')
    ASM = feature.greycoprops(graycom, 'ASM')

    print("Contrast: {}".format(contrast))
    print("Dissimilarity: {}".format(dissimilarity))
    print("Homogeneity: {}".format(homogeneity))
    print("Energy: {}".format(energy))
    print("Correlation: {}".format(correlation))
    print("ASM: {}".format(ASM))
    
    list = []    
    # appending instances to list
    list.append(contrast)
    list.append(dissimilarity)
    list.append(homogeneity)
    list.append(energy)
    list.append(correlation)
    list.append(ASM)
    
    print ("Successfully used GLCM algorithm")
    return list




