import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
data = pd.read_csv("./Dulieudiem.csv")
dt_Train,dt_Test = train_test_split(data,test_size=0.3,shuffle=True)

X_train = dt_Train.iloc[:,[1,2,3,4]]
y_train = dt_Train.iloc[:,5]
X_test = dt_Test.iloc[:,[1,2,3,4]]
y_test = dt_Test.iloc[:,5]
print(f'X_train = \n{X_train}\nY_train = \n{y_train}\nX_test = {X_test}\nY_test = \n{y_test}')

reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
y_test_array = np.array(y_test)
print("Linear Regression")
print("Coeffcient of determination: %.2f" % r2_score(y_test,y_pred))
print("MAE: %.2f" % mean_absolute_error(y_test,y_pred))
MSE = mean_squared_error(y_test,y_pred)
print("MSE: %.2f" % MSE)
RMSE = np.sqrt(MSE)
print('RMSE: %.2f'%RMSE)
print("Thuc te          Du doan              Chenh lech")
for i in range(0,len(y_test_array)):
    print("%.2f" % y_test_array[i]," ",y_pred[i]," ",abs(y_test_array[i]-y_pred[i]))
