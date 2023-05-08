# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 22:01:04 2022

@author: Supun Madushanka
"""

import pandas as pd
from FilesForTraining.MainModel import PreProcessing 
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from FilesForTraining.MainModel.Features.BrownAreaAndCircularity import Brown_Area_And_Circularity
from FilesForTraining.MainModel.Features.YellowAreaAndCircularity import Yellow_Area_And_Circularity
from FilesForTraining.MainModel.Features.WhiteAreaAndCircularity import White_Area_And_Circularity
from FilesForTraining.MainModel.Features.TextureParametersSobel import TextureParametersSobel
from FilesForTraining.MainModel.Features.Hog import HogFilter
from FilesForTraining.MainModel.Features.Sift import SiftFilter
from FilesForTraining.MainModel.Features.Gradient import GradientFeature
from FilesForTraining.MainModel.Features.LaplacianTextureFeatures import LaplacianTextureFeatures
from FilesForTraining.MainModel.Features.Canny import CannyFeatures

def feature_extractor_mask(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  
        
        df = pd.DataFrame()
        img = dataset[image, :,:]
    
    
        #pre processing
        #bg_rem_img = PreProcessing.bg_remove_int(img)
        #smoothen = PreProcessing.image_smooth_int(bg_rem_img)
        #enhance = PreProcessing.improve_contrast_int(smoothen)
        #segmented = PreProcessing.segment_disease_with_range(bg_rem_img, 45, 140)
        
        bg_rem_img = PreProcessing.bg_remove_int(img)
        

        # FEATURE 1 perimeter & circularity of brown
        AreaAndCircularityBrown=Brown_Area_And_Circularity(bg_rem_img)
        
        # FEATURE 2 Laplacian mean and sd
        SobelMeanAndSD=TextureParametersSobel(bg_rem_img)
        
        # FEATURE 3 perimeter & circularity of yellow
        AreaAndCircularityYellow=Yellow_Area_And_Circularity(bg_rem_img)
        
        # FEATURE 3 perimeter & circularity of yellow
        AreaAndCircularityWhite=White_Area_And_Circularity(bg_rem_img)
        

        df = pd.DataFrame({
            'BrownArea1': [AreaAndCircularityBrown[0]], 
            'BrownCircularity1': [AreaAndCircularityBrown[1]],
            'BrownArea2': [AreaAndCircularityBrown[2]], 
            'BrownCircularity2': [AreaAndCircularityBrown[3]],
            'BrownArea3': [AreaAndCircularityBrown[4]], 
            'BrownCircularity3': [AreaAndCircularityBrown[5]],
            'BrownArea4': [AreaAndCircularityBrown[6]], 
            'BrownCircularity4': [AreaAndCircularityBrown[7]],
            'BrownArea5': [AreaAndCircularityBrown[8]], 
            'BrownCircularity5': [AreaAndCircularityBrown[9]],
            'NoOfBrownSpots': [AreaAndCircularityBrown[10]],
            
            'SobelMean': [SobelMeanAndSD[0]],
            'SobelSD': [SobelMeanAndSD[1]],
            'SobelVariance': [SobelMeanAndSD[2]],
            
            'YellowArea1': [AreaAndCircularityYellow[0]], 
            'YellowCircularity1': [AreaAndCircularityYellow[1]],
            'YellowArea2': [AreaAndCircularityYellow[2]], 
            'YellowCircularity2': [AreaAndCircularityYellow[3]],
            'YellowArea3': [AreaAndCircularityYellow[4]], 
            'YellowCircularity3': [AreaAndCircularityYellow[5]],
            'YellowArea4': [AreaAndCircularityYellow[6]], 
            'YellowCircularity4': [AreaAndCircularityYellow[7]],
            'YellowArea5': [AreaAndCircularityYellow[8]], 
            'YellowCircularity5': [AreaAndCircularityYellow[9]],
            'NoOfYellowSpots': [AreaAndCircularityYellow[10]],
            
            'WhiteArea1': [AreaAndCircularityWhite[0]], 
            'WhiteCircularity1': [AreaAndCircularityWhite[1]],
            'WhiteArea2': [AreaAndCircularityWhite[2]], 
            'WhiteCircularity2': [AreaAndCircularityWhite[3]],
            'WhiteArea3': [AreaAndCircularityWhite[4]], 
            'WhiteCircularity3': [AreaAndCircularityWhite[5]],
            'NoOfWhiteSpots': [AreaAndCircularityWhite[10]],
            
            })
        
        #Append features from current image to the dataset
        image_dataset = pd.concat([image_dataset,df])
        print(image_dataset.shape)
        
        
    return image_dataset


def feature_extractor_GLCM(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  
        
        df = pd.DataFrame()
        img = dataset[image, :,:]
    
    
        #pre processing
        #bg_rem_img = PreProcessing.bg_remove_int(img)
        #smoothen = PreProcessing.image_smooth_int(bg_rem_img)
        #enhance = PreProcessing.improve_contrast_int(smoothen)
        
        bg_rem_img = PreProcessing.bg_remove_int(img)
        segmented = PreProcessing.segment_disease_with_range(bg_rem_img, 45, 140)
        # enhance = PreProcessing.improve_contrast_int(segmented)
        
        
        # FEATURE 1 GLCM
        b,g,r = cv2.split(segmented)
        num = 1;
        for color in (r,g,b):
            for pi in (0, np.pi/4, np.pi/2, 3*np.pi / 4):
                GLCM = graycomatrix(color, [3], [pi])       
                GLCM_Energy = graycoprops(GLCM, 'energy')[0]
                GLCM_corr = graycoprops(GLCM, 'correlation')[0]
                GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
                GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]      
                GLCM_contr = graycoprops(GLCM, 'contrast')[0]
                
                df["Energy" + str(num)] = GLCM_Energy
                df["Corr" + str(num)] = GLCM_corr
                df["Diss_sim" + str(num)] = GLCM_diss
                df["Homogen" + str(num)] = GLCM_hom
                df["Contrast" + str(num)] = GLCM_contr
                
                
                
                num += 1
                    
        # FEATURE 2 Mean Green
             
        
        #Add more filters as needed
        # entropy = shannon_entropy(segmented)
        # df['Entropy'] = entropy

        
        #Append features from current image to the dataset
        image_dataset = pd.concat([image_dataset,df])
        print(image_dataset.shape)
        
        
    return image_dataset


def feature_extractor_Hog(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  
        
        img = dataset[image, :,:]
        
        bg_rem_img = PreProcessing.bg_remove_int(img)
        
        
        # FEATURE 1 Hog
        hog_data = HogFilter(bg_rem_img)
        hog_features = hog_data[0]
            
        image_dataset = pd.concat([image_dataset,hog_features])
        print(image_dataset.shape)
        
        
    return image_dataset


def feature_extractor_Sift(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  
        
        img = dataset[image, :,:]
        
        bg_rem_img = PreProcessing.bg_remove_int(img)
        
        
        # FEATURE 1 Sift
        sift_features = SiftFilter(bg_rem_img)
            
        image_dataset = pd.concat([image_dataset,sift_features])
        print(image_dataset.shape)
        
    return image_dataset


def feature_extractor_Custom(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  
        
        df = pd.DataFrame()
        img = dataset[image, :,:]
        
        bg_rem_img = PreProcessing.bg_remove_int(img)
        
        
        # FEATURE 1 GLCM
        b,g,r = cv2.split(bg_rem_img)
        num = 1;
        for color in (r,g,b):
            for pi in (0, np.pi/4, np.pi/2, 3*np.pi / 4):
                GLCM = graycomatrix(color, [3], [pi], symmetric=True, normed=True)       
                
                GLCM_Energy = graycoprops(GLCM, 'energy')[0]
                GLCM_corr = graycoprops(GLCM, 'correlation')[0]
                GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
                GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]      
                GLCM_contr = graycoprops(GLCM, 'contrast')[0]
                
                df["Energy" + str(num)] = GLCM_Energy
                df["Corr" + str(num)] = GLCM_corr
                df["Diss_sim" + str(num)] = GLCM_diss
                df["Homogen" + str(num)] = GLCM_hom
                df["Contrast" + str(num)] = GLCM_contr
                
                num += 1
                
        
        # FEATURE 2 perimeter & circularity of brown
        AreaAndCircularityBrown=Brown_Area_And_Circularity(bg_rem_img)
        
        # FEATURE 3 Laplacian mean and sd
        SobelMeanAndSD=TextureParametersSobel(bg_rem_img)
        
        # FEATURE 4 perimeter & circularity of yellow
        AreaAndCircularityYellow=Yellow_Area_And_Circularity(bg_rem_img)
        
        # FEATURE 5 perimeter & circularity of yellow
        AreaAndCircularityWhite=White_Area_And_Circularity(bg_rem_img)
        
        # FEATURE 6 gradient
        GradientFeatures = GradientFeature(img)
        
        # FEATURE 6 laplacian texture features
        LaplacianFeatures = LaplacianTextureFeatures(img)
        
        # FEATURE 7 canny edge features
        CannyEdgeFeatures = CannyFeatures(img)
        

        df1 = pd.DataFrame({
            'BrownArea1': [AreaAndCircularityBrown[0]], 
            'BrownCircularity1': [AreaAndCircularityBrown[1]],
            'BrownArea2': [AreaAndCircularityBrown[2]], 
            'BrownCircularity2': [AreaAndCircularityBrown[3]],
            'BrownArea3': [AreaAndCircularityBrown[4]], 
            'BrownCircularity3': [AreaAndCircularityBrown[5]],
            'BrownArea4': [AreaAndCircularityBrown[6]], 
            'BrownCircularity4': [AreaAndCircularityBrown[7]],
            'BrownArea5': [AreaAndCircularityBrown[8]], 
            'BrownCircularity5': [AreaAndCircularityBrown[9]],
            'NoOfBrownSpots': [AreaAndCircularityBrown[10]],
            
            'SobelMean': [SobelMeanAndSD[0]],
            'SobelSD': [SobelMeanAndSD[1]],
            'SobelVariance': [SobelMeanAndSD[2]],
            
            'YellowArea1': [AreaAndCircularityYellow[0]], 
            'YellowCircularity1': [AreaAndCircularityYellow[1]],
            'YellowArea2': [AreaAndCircularityYellow[2]], 
            'YellowCircularity2': [AreaAndCircularityYellow[3]],
            'YellowArea3': [AreaAndCircularityYellow[4]], 
            'YellowCircularity3': [AreaAndCircularityYellow[5]],
            'YellowArea4': [AreaAndCircularityYellow[6]], 
            'YellowCircularity4': [AreaAndCircularityYellow[7]],
            'YellowArea5': [AreaAndCircularityYellow[8]], 
            'YellowCircularity5': [AreaAndCircularityYellow[9]],
            'NoOfYellowSpots': [AreaAndCircularityYellow[10]],
            
            'WhiteArea1': [AreaAndCircularityWhite[0]], 
            'WhiteCircularity1': [AreaAndCircularityWhite[1]],
            'WhiteArea2': [AreaAndCircularityWhite[2]], 
            'WhiteCircularity2': [AreaAndCircularityWhite[3]],
            'WhiteArea3': [AreaAndCircularityWhite[4]], 
            'WhiteCircularity3': [AreaAndCircularityWhite[5]],
            'NoOfWhiteSpots': [AreaAndCircularityWhite[10]],
            
            'MeanMagnitude': [GradientFeatures[0]], 
            'SDofMagnitude': [GradientFeatures[1]], 
            'MeanDirection': [GradientFeatures[2]], 
            'SDofDirection': [GradientFeatures[3]],
            
            'Roughness': [LaplacianFeatures[0]], 
            'Smoothness': [LaplacianFeatures[1]], 
            'Coarseness': [LaplacianFeatures[2]], 
            'Regularity': [LaplacianFeatures[3]], 
            'Contrast': [LaplacianFeatures[4]], 
            'Homogeneity': [LaplacianFeatures[5]], 
            'Entropy':  [LaplacianFeatures[6]], 
            'Energy':  [LaplacianFeatures[7]], 
            'Correlation':  [LaplacianFeatures[8]], 
            'Directionality':  [LaplacianFeatures[9]], 
            'FractalDimension': [LaplacianFeatures[10]], 
            
            # 'MeanEdgeIntensity': CannyEdgeFeatures[0],
            # 'StdDevEdgeIntensity': CannyEdgeFeatures[1],
            # 'SkewnessEdgeIntensity': CannyEdgeFeatures[2],
            # 'KurtosisEdgeIntensity': CannyEdgeFeatures[3]
            
            })
        
        new_df = pd.concat([df, df1], axis=1)
        
        image_dataset = pd.concat([image_dataset,new_df])
        print(image_dataset.shape)
        
        
    return image_dataset