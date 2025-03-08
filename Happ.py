import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("House Prediction App")
st.write("Enter the house details to predict the price:")

crim = st.number_input("CRIM: Crime rate per capita", min_value=0.0, step=0.1)
zn = st.number_input("ZN: Residential land zoned for large lots", min_value=0.0, step=0.1)
indus = st.number_input("INDUS: Non-retail business acres per town", min_value=0.0, step=0.1)
chas = st.number_input("CHAS: Charles River dummy variable", min_value=0, max_value=1)
nox = st.number_input("NOX: Nitrogen oxide concentration", min_value=0.0, step=0.1)
rm = st.number_input("RM: Avg number of rooms per dwelling", min_value=0.0, step=0.1)
age = st.number_input("AGE: Owner-occupied units built before 1940", min_value=0.0, step=0.1)
dis = st.number_input("DIS: Distance to employment centers", min_value=0.0, step=0.1)
rad = st.number_input("RAD: Accessibility to highways", min_value=0, step=1)
tax = st.number_input("TAX: Property tax rate per $10,000", min_value=0.0, step=0.1)
ptratio = st.number_input("PTRATIO: Pupil-teacher ratio by town", min_value=0.0, step=0.1)
b = st.number_input("B: Proportion of African American residents", min_value=0.0, step=0.1)
lstat = st.number_input("LSTAT: Percentage of lower-status population", min_value=0.0, step=0.1)

features = np.array([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]).reshape(1, -1)
features_scaled = scaler.transform(features)

if st.button("Predict"):
    try:
        prediction = model.predict(features_scaled)
        st.write(f"Predicted House Price: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.write(f"Error: {e}")
