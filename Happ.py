import numpy as np
import pandas as pd
import joblib
import streamlit as st

model=joblib.load(r'C:\C_py\Python\Scikit\House\models\xgboost_model.joblib')
scaler=joblib.load(r'C:\C_py\Python\Scikit\House\models\scaler.joblib')
st.title("House Prediction App")
st.write("This is an interactive app to predict house prices")
st.write("Please Enter the details Below: ")

crim = st.number_input("CRIM: Crime rate per capita", min_value=0.0, step=0.1)
zn = st.number_input("ZN: Proportion of residential land zoned for large lots", min_value=0.0, step=0.1)
indus = st.number_input("INDUS: Proportion of non-retail business acres per town", min_value=0.0, step=0.1)
chas = st.number_input("CHAS: Charles River dummy variable", min_value=0, max_value=1)
nox = st.number_input("NOX: Nitrogen oxide concentration", min_value=0.0, step=0.1)
rm = st.number_input("RM: Average number of rooms per dwelling", min_value=0.0, step=0.1)
age = st.number_input("AGE: Proportion of owner-occupied units built before 1940", min_value=0.0, step=0.1)
dis = st.number_input("DIS: Weighted distance to employment centers", min_value=0.0, step=0.1)
rad = st.number_input("RAD: Index of accessibility to radial highways", min_value=0, step=1)
tax = st.number_input("TAX: Property tax rate per $10,000", min_value=0.0, step=0.1)
ptratio = st.number_input("PTRATIO: Pupil-teacher ratio by town", min_value=0.0, step=0.1)
b = st.number_input("B: Proportion of residents of African American descent", min_value=0.0, step=0.1)
lstat = st.number_input("LSTAT: Percentage of lower status population", min_value=0.0, step=0.1)

features=np.array([crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]).reshape(1,-1)

features_scaled=scaler.transform(features)


if st.button('Predict'):
    try:
        prediction=model.predict(features_scaled)
        st.write(f"The Predicted Price of the House Price is: ${prediction[0]:.4f}")

    except Exception as e:
        st.write(f"Error Occured: {e}")