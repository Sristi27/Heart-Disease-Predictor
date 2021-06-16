import numpy as np
import pickle
from flask import Flask,request,jsonify,render_template

app=Flask(__name__)
heart_model = pickle.load(open('heart_model.pkl','rb'))


@app.route('/predict',methods=['POST'])
def predict():
    
    features=[float(x) for x in request.form.values()]
    output=heart_model.predict([np.array(features)])
    return output[0]
    

@app.route('/')
def home():
    return "Gender Detection API working perfectly!"
    
if __name__=='main':
    app.run(debug=True)


