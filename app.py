# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:36:03 2023

@author: Supun Madushanka
"""

import numpy as np
from flask import Flask, request, render_template
import pickle


#Create an app object using the Flask class. 
app = Flask(__name__)


#Load the trained main model. (Pickle file)
model = pickle.load(open('models/main_model.pkl', 'rb'))


# Routes

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    # prediction = model.predict(features)  # features Must be in the form [[a, b]]

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Percent with heart disease is {}'.format("HI"))




if __name__ == "__main__":
    app.run()
    