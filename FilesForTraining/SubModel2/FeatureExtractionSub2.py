
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 00:26:15 2022

@author: ishar
"""
import numpy as np 
import cv2
import pandas as pd

from FilesForTraining.SubModel2.featureExtraction.yellowAreaDetector import yellow_mask
from FilesForTraining.SubModel2.featureExtraction.glcm import glcm
from FilesForTraining.SubModel2.preprocessed import preprocesed_image


# FEATURE EXTRACTOR function
# input shape is (n, x, y, c) - number of images, x, y, and channels
def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):  #iterate through each file 
        # print(image)
        
        df = pd.DataFrame()  
        #Reset dataframe to blank after each loop.
        
        input_img = x_train[image, :,:,:]
        img = input_img
        
        ################################################################
        # PREPROCESSING IMAGES
        SIZE = 256
        preprocessed_img = preprocesed_image(img, SIZE)
        img = preprocessed_img
    
    
        ################################################################
        # ADDING DATA TO THE DATAFRAME
            
         # FEATURE 1 - Pixel values
         
        #Add pixel values to the data frame
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values   #Pixel value itself as a feature
        
        # FEATURE 2 - Bunch of Gabor filter responses
        
        num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):   #Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  #Sigma with 1 and 3
                lamda = np.pi/4
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                #print(gabor_label)
                ksize=25
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                
         
        # FEATURE 3 Applied Yellow mask
        
        yellow_area = yellow_mask(img)
        output = yellow_area.reshape(-1)
        df['Yellow_area'] = output
        
        
        
        # FEATURE 4 GLCM applied   
        
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'Energy', 'Correlation', 'ASM']
        angles = ['0', '45', '90','135']
        
        list_feature = glcm(img)

        for i in range (6):
            for j in range (4):
                 column = properties[i] + "_" + angles[j]
                 df[column] = list_feature[i][0][j]
        
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # Split the LAB channels
        L,A,B = cv2.split(lab)

        # Threshold the A channel to isolate the green regions
        ret, threshA = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Threshold the B channel to isolate the yellow regions
        ret, threshB = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Combine the green and yellow regions
        thresh = cv2.bitwise_or(threshA, threshB)

        # Apply morphological operations to refine the segmentation
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Detect the contours of the segmented regions
        contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the regions that exhibit the mottled or mosaic-like pattern
        for cnt in contours:
             area = cv2.contourArea(cnt)
             if area > 500 and area < 5000:
                x,y,w,h = cv2.boundingRect(cnt)
                roi = img[y:y+h, x:x+w]
                cv2.imshow('Region of interest', roi)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        
    return image_dataset


