import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file Excel
df = pd.read_csv('C:/Users/ACER/Desktop/STUDY/hoc_may_1/doanHocmay/house_price_fixed.csv')

# Chọn các đặc trưng và biến mục tiêu
features = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']]
target = df['Price']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xây dựng mô hình hồi quy K-NN
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
# Dự đoán trên tập kiểm tra
y_pred = knn_model.predict(X_test_scaled)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Vẽ đồ thị phân tán
plt.scatter(y_test, y_pred, label='Dự đoán')
# Vẽ đường hồi quy (đường chéo)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Đường hồi quy')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Dự đoán')
plt.title('Giá trị thực tế và Dự đoán cho Giá nhà')
plt.legend()
plt.show()
