import pickle
import os
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app=Flask(__name__)
## load the model
linear_mod=pickle.load(open("Linear_model_boston.pkl",'rb'))
scale_model=pickle.load(open("scaled_transfrom_model.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    ##new_data=np.array(list(data.values())).reshape(1,-1)
    new_data=scale_model.transfrom(np.array(list(data.values())).reshape(1,-1)) ##new data first need to tranfrom
    output=linear_mod.predict(new_data)
    print(output)
if __name__=='__main__':
    app.run(debug=True)    
