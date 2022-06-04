import numpy as np
import pickle
from flask import Flask,request,jsonify,render_template
from flask_cors import CORS

app=Flask(__name__)
heart_model = pickle.load(open('heart_pred_model.pkl','rb'))
CORS(app)

@app.route('/predict',methods=['POST'])
def predict():
    
    features=[]
    features.append(float(request.form['age']))
    features.append(float(request.form['sex']))
    features.append(float(request.form['cp']))
    features.append(float(request.form['trestbps']))
    features.append(float(request.form['chol']))
    features.append(float(request.form['fbs']))
    features.append(float(request.form['restecg']))
    features.append(float(request.form['thalach']))
    features.append(float(request.form['exang']))
    #features.append(float(request.form['oldpeak']))
    #features.append(float(request.form['slope']))
    #features.append(float(request.form['ca']))
    features.append(float(request.form['thal']))
    output=heart_model.predict([np.array(features)])
    if(output[0]==1): value= "Yes"
    else: value= "No"
    return jsonify(msg="Succesfully fetched",value=value, formValues=request.form)
    

@app.route('/',methods=['GET'])
def home():
   return jsonify(msg="Detection API working perfectly!")
    
if __name__=='main':
    app.run(debug=True)


