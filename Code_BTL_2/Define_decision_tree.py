import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('./diabetes_prediction_dataset.csv')

# Lớp Decision tree
class DecisionTree:
    #  Hàm tạo cây quyết định
    def __init__(self):
        self.tree = None

    # Tính Gini index
    def _calculate_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    # Chia dữ liệu
    def _split_data(self, X, y, feature_index, threshold):
        # feature_index : chỉ số đặc trưng của cột dữ liệu
        # Threshold là ngưỡng giá trị muốn sử dụng để chia dữ liệu
        # Khởi tạo 4 danh sách rỗng đẻ chứa dữ liệu khi được chia
        left_X, left_y = [], []
        right_X, right_y = [], []

        # Duyệt qua từng mẫu trong X
        for i, value in enumerate(X[:, feature_index]):
            # Nếu giá trị đặc trưng tại feature_index <= threshold thì thêm mẫu đó và nhãn tương ứng vào left_X và left_y
            if value <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            # Nếu giá trị đặc trưng tại feature_index > threshold thì thêm mẫu đó và nhãn tương ứng vào right_X và right_y
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        #Trả về 4 mảng với các dữ liệu đã được chia
        return np.array(left_X), np.array(left_y), np.array(right_X), np.array(right_y)

    #  Tìm ra đặc trưng và giá trị ngưỡng tốt nhất để chia dữ liệu
    def _find_best_split(self, X, y):
        best_gini = 1
        best_feature_index = -1
        best_threshold = None

        # Duyệt qua các bản ghi
        for feature_index in range(X.shape[1]):

            # Tìm giá trị duy nhất, loại bỏ cac giá trị trùng lặp để dễ dàng lặp
            unique_values = np.unique(X[:, feature_index])

            # Duyệt qua các giá trị duy nhất vừa lấy được
            for value in unique_values:

                # Chia dữ liệu trên giá trị ngưỡng hiện tại của đặc trưng hiện tại
                left_X, left_y, right_X, right_y = self._split_data(X, y, feature_index, value)

                # Tính gini index cho cả 2 nhóm dữ liệu vừa chia
                gini_left = self._calculate_gini(left_y)
                gini_right = self._calculate_gini(right_y)
                gini = (len(left_y) / len(y)) * gini_left + (len(right_y) / len(y)) * gini_right

                # Nếu gini index nhỏ hơn gini index tốt nhất thì trả về chỉ số tốt nhất và kết thúc vòng lặp
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = value

        # trả về chỉ số feature tốt nhất và threshold tootsa nhất
        return best_feature_index, best_threshold

    # Tạo node lá
    def _create_leaf(self, y):
        # Sử dụng hàm np.unique để tìm tất cả các lớp duy nhất trong y và số lượng mỗi lớp. classes chứa các lớp duy nhất và counts chứa số lượng tương ứng của mỗi lớp.
        classes, counts = np.unique(y, return_counts=True)

        # Tìm lớp chiếm đa số bằng cách tìm lớp có số lượng lớn nhất trong counts. majority_class là lớp có số lượng lớn nhất.
        majority_class = classes[np.argmax(counts)]

        # Tạo một dic leaf với hai khóa: 'leaf' và 'class'. 'leaf' được đặt thành True để chỉ ra rằng đây là một nút lá và 'class' được đặt thành majority_class để chỉ ra lớp của nút lá.
        leaf = {'leaf': True, 'class': majority_class}

        # Trả về nút lá
        return leaf

    # Tạo cây quyết định
    def _create_tree(self, X, y):
        # Kiểm tra xem có bao nhiêu lớp duy nhất trong y. Nếu chỉ có một lớp, tạo một nút lá với lớp đó và trả về.
        if len(np.unique(y)) == 1:
            return self._create_leaf(y)

        # Tìm đặc trưng và giá trị ngưỡng tốt nhất để chia dữ liệu bằng cách sử dụng hàm _find_best_split
        feature_index, threshold = self._find_best_split(X, y)

        # Chia dữ liệu thành hai nhóm dựa trên đặc trưng và giá trị ngưỡng tốt nhất bằng cách sử dụng hàm _split_data.
        left_X, left_y, right_X, right_y = self._split_data(X, y, feature_index, threshold)

        # Nếu một trong hai nhóm không có dữ liệu (tức là, tất cả các mẫu đều thuộc về một nhóm), tạo một nút lá với lớp chiếm đa số trong y và trả về.
        if len(left_y) == 0 or len(right_y) == 0:
            return self._create_leaf(y)

        # Nếu cả hai nhóm đều có dữ liệu, tạo một nút không phải lá với thông tin về đặc trưng và giá trị ngưỡng được sử dụng để chia dữ liệu,
        #  cũng như hai nhánh con được tạo bằng cách đệ quy gọi hàm _create_tree trên hai nhóm dữ liệu đã chia.
        tree = {
            'leaf': False,
            'feature_index': feature_index,
            'threshold': threshold,
            'left': self._create_tree(left_X, left_y),
            'right': self._create_tree(right_X, right_y)
        }

        # Trả về cây quyết định đã tạo
        return tree

    # Tìm w
    def fit(self, X, y):
        self.tree = self._create_tree(X, y)

    # Hàm dự đoán mẫu dữ liệu cụ thể
    def _predict_sample(self, sample, tree):
        # Nếu nút hiện tại là nút lá sẽ trả về lớp của nút lá đó
        if tree['leaf']:
            return tree['class']

        # Nếu giá trị đặc trưng tại feature index trong mẫu nhỏ hơn hoặc bằng ngưỡng giá trị nó sẽ gọi đệ quy chính n với nút con bên trái
        if sample[tree['feature_index']] <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])

        # Nếu giá trị của đặc trưng lớn hơn ngưỡng giá trị hàm đệ quy sẽ gọi chính nó với nút con bên phải
        else:
            return self._predict_sample(sample, tree['right'])

    # hàm tìm y dự đoán
    def predict(self, X):
        # Nếu cây quyết định chưa được xây dựng hàm sẽ hiện ra ngoại lệ với thông báo như dưới
        if self.tree is None:
            raise Exception("The model hasn't been trained yet.")
        # Tạo 1 list rỗng để lưu các dự đoán
        predictions = []

        # Nếu mẫu dự đoán cụ thể có trong X nó sẽ dự đoán y dự đoán dựa trên cây quyết định
        for sample in X:
            prediction = self._predict_sample(sample, self.tree)
            # Thêm dự đoán vào list
            predictions.append(prediction)

        # chuyển thành mảng và trả về giá trị
        return np.array(predictions)

# Đọc dữ liệu
X_data = df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']].values
y_data = df['diabetes'].values

# Chia thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=True)

# Sử dụng mô hình DecisionTreeClassifier
clf = DecisionTree()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print("Accuracy: ", accuracy)

precision = precision_score(y_test, y_predict, average='weighted')
print("Precision: ", precision)

recall = recall_score(y_test, y_predict, average='weighted')
print("Recall: ", recall)

f1 = f1_score(y_test, y_predict, average="weighted")
print("F1: ", f1)

