import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns

# Đọc dữ liệu từ DataFrame
df = pd.read_csv('C:/Users/ACER/Desktop/STUDY/hoc_may_1/doanHocmay/house_price_fixed.csv')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xây dựng mô hình Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Dự đoán giá nhà trên tập kiểm thử
y_pred = linear_model.predict(X_test_scaled)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Tính giá trị R2
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Tính residuals
residuals = y_test - y_pred
# Vẽ biểu đồ mô hình hồi quy và residuals
plt.figure(figsize=(12, 6))
# Biểu đồ mô hình hồi quy
plt.subplot(1, 2, 1)
sns.regplot(x=y_pred, y=y_test, scatter_kws={'alpha':0.5})
plt.title('Mô hình hồi quy')
# Biểu đồ residuals
plt.subplot(1, 2, 2)
sns.residplot(x=y_pred, y=residuals, scatter_kws={'alpha':0.5})
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Đường trung bình')
plt.title('Biểu đồ Residuals')
plt.tight_layout()
plt.show()
print(residuals)

