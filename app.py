# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:36:03 2023

@author: Supun Madushanka
"""

import numpy as np
from numpy import genfromtxt
from flask import Flask, request, render_template
import pickle
import cv2
from FilesForTraining.MainModel import FeatureExtraction 

#Create an app object using the Flask class. 
app = Flask(__name__)


#Load the trained main model. (Pickle file)
model = pickle.load(open('models/main_model.pkl', 'rb'))


my_data = genfromtxt('FilesForTraining/MainModel/TestLables.csv', delimiter=',')

from pandas import read_csv
df = read_csv('FilesForTraining/MainModel/TestLables.csv')

df.drop(columns=df.columns[0], axis=1, inplace=True)

df = df.to_numpy()


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df)


# Routes

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    
    img = request.files['image']
    
    image_data = img.read()
    nparr = np.frombuffer(image_data, np.uint8)
    # decode the image data into a numpy array using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #Extract features and reshape to right dimensions
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_features=FeatureExtraction.feature_extractor_Custom(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))

    #Predict
    img_prediction = model.predict(input_img_for_RF)
    img_prediction = le.inverse_transform([img_prediction])  #Reverse the label encoder to original name
        
    print("The prediction for image is: ", img_prediction)
    print("The actual label for image is: Tomato___Early_blight")

    return render_template(
        'index.html', 
        prediction_text='Percent with heart disease is {}'.format(img_prediction),
        prediction_text1='hi'
        )




if __name__ == "__main__":
    app.run()
    