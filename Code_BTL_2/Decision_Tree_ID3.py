import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
df = pd.read_csv('./diabetes_prediction_dataset.csv')

# Chuyển đổi dữ liệu chuỗi sang số
def data_encoder(X):
    label_encoder = LabelEncoder()

    for i in range(X.shape[1]):
        X[:, i] = label_encoder.fit_transform(X[:, i])
    return X

# Đọc dữ liệu và chuyển đổi
X_data = np.array(df[['gender', 'age', 'hypertension','heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level','diabetes']].values)
X_data = data_encoder(X_data)

# Chia thành tập huấn luyện và tập kiểm tra
dt_Train, dt_Test = train_test_split(X_data, test_size=0.3, shuffle=True)

# Tách dữ liệu và nhãn
X_train = dt_Train[:, :8]
Y_train = dt_Train[:, 8].astype(int)  # đổi kiểu của tập nhãn từ object thành số nguyên
X_test = dt_Test[:, :8]
y_test = dt_Test[:, 8].astype(int)   # đổi kiểu của tập nhãn từ object thành số nguyên


# Sử dụng mô hình Decision Tree
dcf = DecisionTreeClassifier(criterion='entropy', splitter='best')
dcf_fit = dcf.fit(X_train,Y_train)
y_predict = dcf.predict(X_test)

accuracy = accuracy_score(y_test,y_predict)
print("Accuracy: ",accuracy)

precision = precision_score(y_test,y_predict,average='weighted')
print("Precision: ",precision)

recall = recall_score(y_test,y_predict,average='weighted')
print("Recall: ",recall)

f1 = f1_score(y_test,y_predict,average="weighted")
print("F1: ",f1)

