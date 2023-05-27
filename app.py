# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:36:03 2023

@author: Supun Madushanka
"""

import numpy as np
from flask import Flask, request, render_template, url_for
import pickle
import cv2
from FilesForTraining.MainModel import FeatureExtractionMain 
from FilesForTraining.SubModel1 import FeatureExtractionSub1 
from FilesForTraining.SubModel2 import FeatureExtractionSub2
from FilesForTraining.SubModel3 import FeatureExtractionSub3
from FilesForTraining.SubModel3 import PreprocessingSub3
import os

#Create an app object using the Flask class. 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'


# Main Model----------------------------------------------------------
#Load the trained main model. (Pickle file)
model = pickle.load(open('models/main_model.pkl', 'rb'))

from pandas import read_csv
df = read_csv('FilesForTraining/MainModel/TestLables.csv')

df.drop(columns=df.columns[0], axis=1, inplace=True)

df = df.to_numpy()


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df)
#---------------------------------------------------------------------




# Sub Model 1----------------------------------------------------------
#Load the trained main model. (Pickle file)
model1 = pickle.load(open('models/class_1_model.pkl', 'rb'))

from pandas import read_csv
df1 = read_csv('FilesForTraining/SubModel1/TestLables.csv')

df1.drop(columns=df1.columns[0], axis=1, inplace=True)

df1 = df1.to_numpy()


from sklearn import preprocessing
le1 = preprocessing.LabelEncoder()
le1.fit(df1)
#---------------------------------------------------------------------



# Sub Model 2----------------------------------------------------------
#Load the trained main model. (Pickle file)
model2 = pickle.load(open('models/class_2_model.pkl', 'rb'))

from pandas import read_csv
df2 = read_csv('FilesForTraining/SubModel2/TestLables.csv')

df2.drop(columns=df2.columns[0], axis=1, inplace=True)

df2 = df2.to_numpy()


from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()
le2.fit(df2)
#---------------------------------------------------------------------



# Sub Model 3----------------------------------------------------------
#Load the trained main model. (Pickle file)
model3 = pickle.load(open('models/class_3_model.pkl', 'rb'))

from pandas import read_csv
df3 = read_csv('FilesForTraining/SubModel3/TestLables.csv')

df3.drop(columns=df3.columns[0], axis=1, inplace=True)

df3 = df3.to_numpy()


from sklearn import preprocessing
le3 = preprocessing.LabelEncoder()
le3.fit(df3)
#---------------------------------------------------------------------



# Routes

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    categories = 'categories.html'
    footer = 'footer.html'
    return render_template(
        'index.html',
        categories=categories,
        footer=footer
        )



@app.route('/predict',methods=['POST'])
def predict():
    
    categories = 'categories.html'
    footer = 'footer.html'
    yellowLeafCurl = 'yellow-leaf-curl.html'
    earlyBlight = 'early-blight.html'
    lateBlight = 'late-blight.html'
    leafMold = 'leaf-mold.html'
    mosaicVirus = 'mosaic-virus.html'
    spiderMites = 'spider-mites.html'
    bacterialSpot = 'bacterial-spot.html'
    septoriaLeafSpot = 'septoria-leaf-spot.html'
    powderyMildew = 'powdery-mildew.html'
    healthy = 'healthy.html'
    
    
    img = request.files['image']
    
    image_data = img.read()
    nparr = np.frombuffer(image_data, np.uint8)
    # decode the image data into a numpy array using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #Extract features and reshape to right dimensions
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_features=FeatureExtractionMain.feature_extractor_Custom(input_img)
    
    # remove nan values by 0 and infinite values by large value
    input_img_features[np.isnan(input_img_features)] = 0
    
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))

    # remove infinite values by large value
    large_num = 1e9
    input_img_for_RF[np.isinf(input_img_for_RF)] = np.sign(input_img_for_RF[np.isinf(input_img_for_RF)]) * large_num
    
    #Predict
    img_prediction = model.predict(input_img_for_RF)
    img_prediction = le.inverse_transform([img_prediction])  #Reverse the label encoder to original name
        
    print("-----The prediction for image is: ", img_prediction[0])
    
    
    if(img_prediction[0]=='class_1'):
        #Extract features and reshape to right dimensions
        input_img_features_sub_1=FeatureExtractionSub1.feature_extractor_custom(input_img)
        input_img_features_sub_1 = np.expand_dims(input_img_features_sub_1, axis=0)
        input_img_for_RF_sub_1 = np.reshape(input_img_features_sub_1, (input_img.shape[0], -1))

        #Predict
        img_prediction_sub_1 = model1.predict(input_img_for_RF_sub_1)
        img_prediction_sub_1 = le1.inverse_transform([img_prediction_sub_1])  #Reverse the label encoder to original name
            
        print("-----The prediction for image is: ", img_prediction_sub_1)
        
        # img_path = os.path.join(app.config['UPLOAD_FOLDER'], copy_img.filename)
        # copy_img.save(img_path)
        
        return render_template(
            'index.html', 
            prediction_category=img_prediction[0],
            prediction_disease=img_prediction_sub_1[0],
            prediction_text='Disease category is {}'.format(img_prediction[0]),
            prediction_text1='Disease prediction is {}'.format(img_prediction_sub_1[0]),
            scroll_to='scroll_here',
            categories=categories,
            footer=footer,
            yellowLeafCurl=yellowLeafCurl,
            earlyBlight=earlyBlight,
            lateBlight=lateBlight,
            leafMold=leafMold,
            mosaicVirus=mosaicVirus,
            spiderMites=spiderMites,
            bacterialSpot=bacterialSpot,
            septoriaLeafSpot=septoriaLeafSpot,
            powderyMildew=powderyMildew,
            healthy=healthy
            )
    elif (img_prediction[0]=='class_2'):
        #Extract features and reshape to right dimensions
        input_img_features_sub_2=FeatureExtractionSub2.feature_extractor(input_img)
        input_img_features_sub_2 = np.expand_dims(input_img_features_sub_2, axis=0)
        input_img_for_RF_sub_2 = np.reshape(input_img_features_sub_2, (input_img.shape[0], -1))

        #Predict
        img_prediction_sub_2 = model2.predict(input_img_for_RF_sub_2)
        img_prediction_sub_2 = le2.inverse_transform([img_prediction_sub_2])  #Reverse the label encoder to original name
            
        print("-----The prediction for image is: ", img_prediction_sub_2)
        
        # img_path = os.path.join(app.config['UPLOAD_FOLDER'], copy_img.filename)
        # copy_img.save(img_path)
        
        return render_template(
            'index.html', 
            prediction_category=img_prediction[0],
            prediction_disease=img_prediction_sub_2[0],
            prediction_text='Disease category is {}'.format(img_prediction[0]),
            prediction_text1='Disease prediction is {}'.format(img_prediction_sub_2[0]),
            scroll_to='scroll_here',
            categories=categories,
            footer=footer,
            yellowLeafCurl=yellowLeafCurl,
            earlyBlight=earlyBlight,
            lateBlight=lateBlight,
            leafMold=leafMold,
            mosaicVirus=mosaicVirus,
            spiderMites=spiderMites,
            bacterialSpot=bacterialSpot,
            septoriaLeafSpot=septoriaLeafSpot,
            powderyMildew=powderyMildew,
            healthy=healthy
            )
    elif (img_prediction[0]=='class_3'):
        desired_size = (128, 128)
        input_img3 = cv2.resize(img, desired_size)
        input_img3=PreprocessingSub3.pre_processor(input_img3)

        input_img3 = np.expand_dims(input_img3, axis=0)
        
        #Extract features and reshape to right dimensions
        input_img_features_sub_3=FeatureExtractionSub3.feature_extractor(input_img3)
        input_img_features_sub_3 = np.expand_dims(input_img_features_sub_3, axis=0)
        input_img_for_RF_sub_3 = np.reshape(input_img_features_sub_3, (input_img3.shape[0], -1))

        #Predict
        img_prediction_sub_3 = model3.predict(input_img_for_RF_sub_3)
        print("-----The prediction for image is: ", img_prediction_sub_3)
        img_prediction_sub_3 = le3.inverse_transform([img_prediction_sub_3])  #Reverse the label encoder to original name
            
        print("-----The prediction for image is: ", img_prediction_sub_3)
        
        # img_path = os.path.join(app.config['UPLOAD_FOLDER'], copy_img.filename)
        # copy_img.save(img_path)
        
        return render_template(
            'index.html', 
            prediction_category=img_prediction[0],
            prediction_disease=img_prediction_sub_3[0],
            prediction_text='Disease category is {}'.format(img_prediction[0]),
            prediction_text1='Disease prediction is {}'.format(img_prediction_sub_3[0]),
            scroll_to='scroll_here',
            categories=categories,
            footer=footer,
            yellowLeafCurl=yellowLeafCurl,
            earlyBlight=earlyBlight,
            lateBlight=lateBlight,
            leafMold=leafMold,
            mosaicVirus=mosaicVirus,
            spiderMites=spiderMites,
            bacterialSpot=bacterialSpot,
            septoriaLeafSpot=septoriaLeafSpot,
            powderyMildew=powderyMildew,
            healthy=healthy
            )
    elif (img_prediction[0]=='class_4'):
        return render_template(
            'index.html', 
            prediction_category=img_prediction[0],
            prediction_disease='Healthy',
            prediction_text='Percent with heart disease is {}'.format(img_prediction[0]),
            prediction_text1='Unknown',
            categories=categories,
            footer=footer,
            yellowLeafCurl=yellowLeafCurl,
            earlyBlight=earlyBlight,
            lateBlight=lateBlight,
            leafMold=leafMold,
            mosaicVirus=mosaicVirus,
            spiderMites=spiderMites,
            bacterialSpot=bacterialSpot,
            septoriaLeafSpot=septoriaLeafSpot,
            powderyMildew=powderyMildew,
            healthy=healthy
            )




if __name__ == "__main__":
    app.run()
    