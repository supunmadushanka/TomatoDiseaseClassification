
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 00:26:15 2022

@author: ishar
"""
import numpy as np 
import cv2
import pandas as pd
from skimage.feature import hog


###################################################################
# FEATURE EXTRACTOR function
def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):  #iterate through each file 
        #print(image)
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        
        input_img = x_train[image, :,:,:]
        img = input_img
    ################################################################
    #START ADDING DATA TO THE DATAFRAME
            
         # FEATURE 1 - Pixel values
         
        #Add pixel values to the data frame
        r, g, b = cv2.split(img)
        pixel_values_blue = b.reshape(-1)
        pixel_values_green = g.reshape(-1)
        pixel_values_red = r.reshape(-1)
        df['Pixel_Value_blue'] = pixel_values_blue 
        df['Pixel_Value_green'] = pixel_values_green
        df['Pixel_Value_red'] = pixel_values_red#Pixel value itself as a feature
        #df['Image_Name'] = image   #Capture image name as we read multiple images
        
        # FEATURE 2 - HOG filter values
        #creating hog features 
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                            cells_per_block=(2, 2), visualize=True, multichannel=True)
        
        #Add pixel values to the data frame
        HOG_pixel_values = hog_image.reshape(-1)
        df['HOG_Value'] = HOG_pixel_values   
        
        # FEATURE 3 - Hue values
        hsv_img= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_img)
        Hue_pixel_values = s.reshape(-1)
        df['Saturation_Value'] = Hue_pixel_values 
        
        # FEATURE 4 - Bunch of Gabor filter responses
        
        num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):   #Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  #Sigma with 1 and 3
                lamda = np.pi/4
                gamma = 0.8
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                ksize=3
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(hsv_img, cv2.CV_8UC3, kernel)
                h2, s2, v2 = cv2.split(fimg)
                gabor_saturation = s2.reshape(-1)
                df[gabor_label+'_sat'] =  gabor_saturation
                #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
        
        
       
        #Add more filters as needed
        
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        
    return image_dataset


