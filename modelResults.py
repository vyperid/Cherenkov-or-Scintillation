#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:52:36 2024

@author: dilarakizilkaya
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifierCV, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
import pandas as pd
import csv
import sys

if len(sys.argv) != 3:
    print("Usage: python your_script.py <input_file> <output_file>")
    exit(1)
 
   
txt_file_read = sys.argv[1]
txt_file_write = sys.argv[2]   

f = open(txt_file_write, "w")
 
dropParam = int(input("How many parameters will you drop? "))
paramList = []


for i in range(2, dropParam+2):
    param_name = input("Enter the parameter name: ")
    paramList.append(param_name) 
    
vectorName = input("What is your y column? ")


data = pd.read_csv(txt_file_read)

for i in range(dropParam):
    X = data.drop(paramList[i], axis=1)
    y = data[vectorName]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ("AdaBoostClassifier", AdaBoostClassifier()),
    ("XGBClassifier", xgb.XGBClassifier()),
    ("RidgeClassifierCV", RidgeClassifierCV()),
    ("RidgeClassifier", RidgeClassifier()),
    ("RandomForestClassifier", RandomForestClassifier()),
    ("ExtraTreesClassifier", ExtraTreesClassifier()),
    ("NearestCentroid", NearestCentroid()),
    ("LogisticRegression", LogisticRegression()),
    ("LinearSVC", LinearSVC()),  
    ("KNeighborsClassifier", KNeighborsClassifier()),
    ("ExtraTreesClassifier", ExtraTreesClassifier()),
    ("ExtraTreeClassifier", ExtraTreeClassifier()),
    ("DecisionTreeClassifier", DecisionTreeClassifier()),
    ("BernoulliNB", BernoulliNB()),
    ("LGBMClassifier", LGBMClassifier())
]

best_f1_score = 0
best_accuracy = 0
best_model_name = ""

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    accuracy = report['accuracy']
    f.write(f"{name}: F1 Score: {f1_score}, Accuracy: {accuracy} \n")
    #print(f"{name}: F1 Score: {f1_score}, Accuracy: {accuracy}")
    if f1_score > best_f1_score and accuracy > best_accuracy:
        best_f1_score = f1_score
        best_accuracy = accuracy
        best_model_name = name

f.write(f"\nModel that has the highest F1 Score ve Accuracy: {best_model_name} (F1 Score: {best_f1_score}, Accuracy: {best_accuracy})")
#print(f"\nModel that has the highest F1 Score ve Accuracy: {best_model_name} (F1 Score: {best_f1_score}, Accuracy: {best_accuracy})")   

f.close()



