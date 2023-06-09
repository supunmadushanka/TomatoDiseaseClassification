# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 22:01:04 2022

@author: Supun Madushanka
"""

import pandas as pd
from FilesForTraining.SubModel1 import PreProcessing

from FilesForTraining.SubModel1.Features.BrownAreaAndCircularity import Brown_Area_And_Circularity
from FilesForTraining.SubModel1.Features.YellowAreaAndCircularity import Yellow_Area_And_Circularity
from FilesForTraining.SubModel1.Features.TextureParametersSobel import TextureParametersSobel
from FilesForTraining.SubModel1.Features.Gradient import GradientFeature
from FilesForTraining.SubModel1.Features.LaplacianTextureFeatures import LaplacianTextureFeatures



def feature_extractor_custom(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  
        
        df = pd.DataFrame()
        img = dataset[image, :,:]
    
    
        #pre processing
        # bg_rem_img = PreProcessing.bg_remove_int(img)
        # smoothen = PreProcessing.image_smooth_int(bg_rem_img)
        # bg_rem_img = smoothen
        #enhance = PreProcessing.improve_contrast_int(smoothen)
        #segmented = PreProcessing.segment_disease_with_range(bg_rem_img, 45, 140)
        
        bg_rem_img = PreProcessing.bg_remove_int(img)
        
        
        
        # FEATURE 1 perimeter & circularity of brown
        AreaAndCircularityBrown=Brown_Area_And_Circularity(bg_rem_img)
        
        # FEATURE 2 Laplacian mean and sd
        SobelMeanAndSD=TextureParametersSobel(bg_rem_img)
        
        # FEATURE 3 perimeter & circularity of yellow
        AreaAndCircularityYellow=Yellow_Area_And_Circularity(bg_rem_img)
        
        # FEATURE 4 gradient
        GradientFeatures = GradientFeature(img)
        
        # FEATURE 5 laplacian texture features
        LaplacianFeatures = LaplacianTextureFeatures(img)
        

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
            
            })
        
        
        #Append features from current image to the dataset
        image_dataset = pd.concat([image_dataset,df])
        print(image_dataset.shape)
        
        
    return image_dataset