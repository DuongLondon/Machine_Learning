import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

data = pd.read_csv("./Dulieudiem.csv")
dt_Train,dt_Test = train_test_split(data,test_size=0.3,shuffle=True)

X_train = dt_Train.iloc[:,[1,2,3,4]]
y_train = dt_Train.iloc[:,5]
X_test = dt_Test.iloc[:,[1,2,3,4]]
y_test = dt_Test.iloc[:,5]

def error(y,y_pred):
    sum=0
    for i in range(0,len(y)):
        sum = sum+abs(y[i]-y_pred[i])
    return sum/len(y)
k=5
kf = KFold(n_splits=k, random_state=None)
max_reg = 999999
max_ls = 999999
max_rg = 999999
i_reg = 1
i_ls = 1
i_rg = 1
for train_index, test_index in kf.split(dt_Train):
    X_train = dt_Train.iloc[train_index,[1,2,3,4]]
    y_train = dt_Train.iloc[train_index,5]
    X_val = dt_Train.iloc[test_index,[1,2,3,4]]
    y_val = dt_Train.iloc[test_index,5]

    y_train_array = np.array(y_train)
    y_val_array = np.array(y_val)

    # Linear Regression
    model_reg = LinearRegression()
    model_reg.fit(X_train,y_train)
    y_pred_train = model_reg.predict(X_train)
    y_pred_val = model_reg.predict(X_val)

    # Lasso
    model_ls = Lasso(alpha=1.0, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
    model_ls.fit(X_train,y_train,sample_weight=None,check_input=True)
    y_pred_ls_train = model_ls.predict(X_train)
    y_pred_ls_val = model_ls.predict(X_val)

    # Ridge
    model_rg = Ridge(alpha=0.1, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)
    model_rg.fit(X_train,y_train)
    y_pred_rg_train = model_rg.predict(X_train)
    y_pred_rg_val  = model_rg.predict(X_val)

    sum_reg = error(y_train_array,y_pred_train) + error(y_val_array,y_pred_val)
    sum_ls = error(y_train_array,y_pred_ls_train) + error(y_val_array,y_pred_ls_val)
    sum_rg = error(y_train_array,y_pred_rg_train) + error(y_val_array,y_pred_rg_val)

    if(sum_reg < max_reg):
        max = sum
        last = i_reg
        best_model_reg = model_reg # Lay ra mo hinh tot nhat
    i_reg = i_reg+1

    if (sum_ls < max_ls):
        max_ls = sum_ls
        last_ls = i_ls
        best_model_ls = model_ls  # Lay ra mo hinh tot nhat
    i_ls = i_ls+1

    if (sum_rg < max_rg):
        max_rg = sum_rg
        last_rg = i_rg
        best_model_rg = model_rg  # Lay ra mo hinh tot nhat
    i_rg = i_rg+1

y_pred_reg = best_model_reg.predict(X_test)
y_pred_ls = best_model_ls.predict(X_test)
y_pred_rg = best_model_rg.predict(X_test)

print("Coefficient of determination - Linear Regression: %.4f" % r2_score(y_test,y_pred_reg))
# print("Thuc te          Du doan              Chenh lech")
# for i in range(0,len(y_test_array)):
#     print("%.2f" % y_test_array[i]," ",y_pred_reg[i]," ",abs(y_test_array[i]-y_pred_reg[i]))

print("Coefficient of determination - Lasso: %.4f" % r2_score(y_test,y_pred_ls))
# print("Thuc te          Du doan              Chenh lech")
# for i in range(0,len(y_test_array)):
#     print("%.2f" % y_test_array[i]," ",y_pred_ls[i]," ",abs(y_test_array[i]-y_pred_ls[i]))

print("Coefficient of determination - Ridge: %.4f" % r2_score(y_test,y_pred_rg))
# print("Thuc te          Du doan              Chenh lech")
# for i in range(0,len(y_test_array)):
#     print("%.2f" % y_test_array[i]," ",y_pred_rg[i]," ",abs(y_test_array[i]-y_pred_rg[i]))
