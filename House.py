import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "data", "boston.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

data = pd.read_csv(DATA_PATH)
x = data.drop(columns="MEDV")
y = data['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)

scaler = StandardScaler()
x_train_sca = scaler.fit_transform(x_train)
x_test_sca = scaler.transform(x_test)

param = {
    'n_estimators': [50, 100, 200],
    'max_depth': [1, 5, 10],
    'learning_rate': [0.01, 0.1, 1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

model = GridSearchCV(estimator=XGBRegressor(), cv=5, verbose=1, param_grid=param, n_jobs=-1)
model.fit(x_train_sca, y_train)

joblib.dump(model.best_estimator_, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

y_pred = model.predict(x_test_sca)
print(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred,)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
