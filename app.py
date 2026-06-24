import streamlit as st
import pickle
import numpy as np
import pandas as pd


# -----------------------------
# Load Model
# -----------------------------

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# -----------------------------
# App Title
# -----------------------------

st.title("Machine Learning Prediction App")

st.write(
    "Enter the values below to get a prediction from the trained model."
)


# -----------------------------
# User Inputs
# CHANGE THESE BASED ON YOUR DATASET
# -----------------------------

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    data = np.array(
        [
            feature1,
            feature2,
            feature3,
            feature4
        ]
    ).reshape(1, -1)


    prediction = model.predict(data)


    if prediction[0] == 1:
        st.success("Prediction: Class 1")
    else:
        st.error("Prediction: Class 0")
      
