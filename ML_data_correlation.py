import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
# Đọc dữ liệu từ file Excel
df = pd.read_csv('C:/Users/ACER/Desktop/STUDY/hoc_may_1/doanHocmay/house_price_fixed.csv')

# Vẽ biểu đồ tương quan bằng heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Biểu đồ Tương quan giữa các Thuộc tính')
plt.show()