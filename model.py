import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

heart_data=pd.read_csv('/Users/sristichowdhury/Desktop/heart.csv')


print(heart_data.head())

print(heart_data.shape)

Y=heart_data['target']
X=heart_data.drop(columns='target',axis=1) #dropping columns


#Splitting Data
train_X,test_X,train_y,test_y=train_test_split(X,Y,random_state=2,test_size=0.2,
                                               stratify=Y)


print(X.shape,train_X.shape,test_X.shape)


#Building and training model

model = LogisticRegression()

model.fit(train_X,train_y)

#saving model to disk
pickle.dump(model,open('heart_model.pkl','wb'))

#predictedValue=model.predict(test_X)
#score=accuracy_score(predictedValue,test_y)

#print('Accuracy score:{}'.format(score))


#For external data
#input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)
#array_input_data=np.asarray(input_data)

#reshape the array as we are predicting for only on instance
#input_data_reshaped=array_input_data.reshape(1,-1) #converting to 2d array

#prediction=model.predict(input_data_reshaped)
#print(prediction)
# //prediction is given in list format 



model = pickle.load(open('heart_model.pkl','rb'))
prediction=model.predict([[41,0,1,130,204,0,0,172,0,1.4,2,0,2]])

if(prediction[0]==0): print("No heart disease")
else: print("Heart Disease")


