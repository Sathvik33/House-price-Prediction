import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,r2_score,mean_squared_error,root_mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso,Ridge
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\C_py\Python\Scikit\House\data\boston.csv")
x=data.drop(columns="MEDV")
y=data['MEDV']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=60)

scaler=StandardScaler()
x_train_sca=scaler.fit_transform(x_train)
x_test_sca=scaler.transform(x_test)
param={
    'n_estimators':[50,100,200],
    'max_depth':[1,5,10],
    'learning_rate':[0.01,0.1,1],
    'subsample':[0.7,0.8,1.0],
    'colsample_bytree':[0.7,0.8,1.0]
}

model=GridSearchCV(estimator=XGBRegressor(),cv=5,verbose=1,param_grid=param,n_jobs=-1)
model.fit(x_train_sca,y_train)

# sns.scatterplot(y,color='blue')
# plt.show()

import joblib
joblib.dump(model.best_estimator_,r'C:\C_py\Python\Scikit\House\xgboost_model.joblib')
joblib.dump(scaler,r'C:\C_py\Python\Scikit\House\scaler.joblib')

y_pred=model.predict(x_test_sca)
print(pd.DataFrame({"Actual":y_test,"predicted":y_pred}))
print(f"Mean squared error: {np.sqrt(mean_squared_error(y_test,y_pred))}")
print(f"Root mean squared error: {root_mean_squared_error(y_test,y_pred)}")
print(f"R2 score: {r2_score(y_test,y_pred)}")
print(f"Mean absolute error: {mean_absolute_error(y_test,y_pred)}")