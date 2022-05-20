import math

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error as mae

from sklearn.ensemble import RandomForestRegressor

result3 = pd.read_csv(r"C:\Users\user\Desktop\Finalstock2.csv")

x = result3[['Sumscore']].values
y = result3[['Open']].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)




regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x_train, y_train)

y_pred_train = regressor.predict(x_train)
y_pred_test = regressor.predict(x_test)

y_pred_train= pd.DataFrame(y_pred_train)
y_train= pd.DataFrame(y_train)
result = pd.concat([y_pred_train,y_train], axis=1)

result.to_csv(r'C:\Users\user\Desktop\y_pred_train.csv',index=False)

rmse = sqrt(mean_squared_error(y_train, y_pred_train ))
print('RMSE=',rmse)
mse = mean_squared_error(y_train,y_pred_train )
print('MSE=',mse)
mae = mae(y_train, y_pred_train)
print('MAE=',mae)

predict_result = regressor.predict(x)
result3=result3.loc[:,['date','Sumscore','Open']]

predict_result = pd.DataFrame(predict_result )
predict_result.columns=['Predict_Open']
GRAPH = pd.concat([result3,predict_result], axis=1)
print(GRAPH)
GRAPH.to_csv(r'C:\Users\user\Desktop\GRAPH.csv', index=False)




