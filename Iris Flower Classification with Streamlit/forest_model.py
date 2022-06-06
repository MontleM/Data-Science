import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('IRIS.csv')
df.head()

# Renaming the target column into numbers to aid training of the model
df['species']= df['species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
  
# splitting the data into the columns which need to be trained(X) and the target column(y)
X = df.drop('species', axis=1)
y = df['species']
  
# splitting data into training and testing data with 20 % of data as testing data respectively
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
  
# importing the random forest classifier model and training it on the dataset
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier()
forest_classifier.fit(X_train, y_train)
  
# predicting on the test dataset
y_pred = forest_classifier.predict(X_test)
  
# finding out the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
score = score * 100

st.write(f'accuracy score of the model is {score:.2f}%')


# pickling the model
import pickle
pickle_out = open("Random_Forest_Classifier.pkl", "wb")
pickle.dump(forest_classifier, pickle_out)
pickle_out.close()
