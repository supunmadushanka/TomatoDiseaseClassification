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



# Routes

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    
    img = request.files['image']
    
    copy_img = img
    
    image_data = img.read()
    nparr = np.frombuffer(image_data, np.uint8)
    # decode the image data into a numpy array using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #Extract features and reshape to right dimensions
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_features=FeatureExtractionMain.feature_extractor_Custom(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))

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
            scroll_to='scroll_here'
            # img_path=img_path
            )
    else:
        print("Not class")
        
        # img_path = os.path.join(app.config['UPLOAD_FOLDER'], copy_img.filename)
        # copy_img.save(img_path)
        
        return render_template(
            'index.html', 
            prediction_category=img_prediction[0],
            prediction_disease='Unknown',
            prediction_text='Percent with heart disease is {}'.format(img_prediction[0]),
            prediction_text1='Unknown',
            # img_path=img_path
            )




if __name__ == "__main__":
    app.run()
    