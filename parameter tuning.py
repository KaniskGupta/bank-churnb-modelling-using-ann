# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 04:43:17 2018

@author: shanu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Churn_Modelling.csv')

x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])

labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier=Sequential()

    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

    
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier)

parameters={'batch_size':[25,32],
            'nb_epoch':[100,500],
            'optimizer':['adam','rmsprop']
        }

grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)

grid_search=grid_search.fit(x_train,Y_train)

best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_ 

 
