# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

model_svc = SVC(kernel="linear").fit(X_train, y_train)
model_lr = LogisticRegression().fit(X_train, y_train)
model_rfc = RandomForestClassifier().fit(X_train, y_train)

model_svc_score = model_svc.score(X_train, y_train)
model_lr_score = model_lr.score(X_train, y_train)
model_rfc_score = model_rfc.score(X_train, y_train)

# Create a function that accepts an ML mode object say 'model' and the four features of an Iris flower as inputs and returns its name.
@st.cache()
def prediction(_model, sl, sw, pl, pw):
  pred = _model.predict(np.array([sl, sw, pl, pw]).reshape(1,-1))
  if pred[0] == 0:
    return "Iris-setosa"
  elif pred[0] == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"

# Add title widget
st.sidebar.title("Iris Flower Species Prediction App")
sl = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
sw = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
pl = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
pw = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))
model = st.sidebar.selectbox("Classifier", ("SVC", "Logistic Regression", "Random Forest Classifier"))
if st.sidebar.button("Predict"):
  if model == "SVC":
    st.write(f"The species is {prediction(model_svc, sl, sw, pl, pw)}")
    st.write(f"The accuracy of the model is {model_svc_score}")
  if model == "Logistic Regression":
    st.write(f"The species is {prediction(model_lr, sl, sw, pl, pw)}")
    st.write(f"The accuracy of the model is {model_lr_score}")
  if model == "Random Forest Classifier":
    st.write(f"The species is {prediction(model_rfc, sl, sw, pl, pw)}")
    st.write(f"The accuracy of the model is {model_rfc_score}")
