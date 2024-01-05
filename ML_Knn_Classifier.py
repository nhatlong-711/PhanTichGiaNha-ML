import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file Excel
df=pd.read_csv('C:/Users/ACER/Desktop/STUDY/hoc_may_1/doanHocmay/house_price_fixed.csv') 

# Chia cột 'Price' thành 3 mức 0, 1, 2
price_bins = [0, 250000, 350000, 450000]
price_labels = [0, 1, 2]
df['Price_Category'] = pd.cut(df['Price'], bins=price_bins, labels=price_labels)

# Chọn các đặc trưng để sử dụng trong mô hình KNN
features = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, df['Price_Category'], test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xây dựng mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn_model.predict(X_test_scaled)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Trực quan hóa kết quả
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=price_labels, yticklabels=price_labels)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn biểu thị giá nhà')
# plt.show()
