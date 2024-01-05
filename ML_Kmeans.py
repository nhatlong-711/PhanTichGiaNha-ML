import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import time
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score

#đọc dữ liệu
doc = pd.read_csv("C:/Users/ACER/Desktop/STUDY/hoc_may_1/doanHocmay/air_fixed.csv")
print(doc)

#chon du lieu
data = ['PM 2.5','temp','pressure','humidity','wind_speed'] 
features = doc[data]
#chọn k phù hợp bằng elbow methods
# Chọn số lượng cụm K
k_values = range(1, 11)
sse_values = []
fit_times = []

# Thực hiện K-means cho mỗi giá trị K và tính SSE và thời gian thực hiện
for k in k_values:
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    fit_time = time.time() - start_time
    sse_values.append(kmeans.inertia_)  # inertia_ chính là SSE
    fit_times.append(fit_time)


# Vẽ đồ thị Elbow và đồ thị thời gian thực hiện K-means
fig, ax1 = plt.subplots()

# Đồ thị Elbow
color = 'tab:red'
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Sum of Squared Errors (SSE)', color=color)
ax1.plot(k_values, sse_values, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Tạo đồ thị thời gian thực hiện K-means trên cùng trục x
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Fit Time (s)', color=color)
ax2.plot(k_values, fit_times, marker='s', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Phương pháp Elbow và Thời gian thực hiện cho K tối ưu')
plt.show()

#du dung Kmeans de phan cum du lieu
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(features)
# lay cac diem trong tam lam cum
centroids = kmeans.cluster_centers_
# Lấy nhãn của từng điểm dữ liệu
labels = kmeans.labels_

# Visualize pairplot để xem sự phân bố và tương quan giữa các biến
sns.pairplot(features)
plt.title('Biểu đồ Pairplot của Các Biến Được Chọn')
plt.show()

# Thực hiện K-means và visualize kết quả
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(features)

# Thêm nhãn cụm vào DataFrame
doc['Cluster'] = kmeans.labels_

# Visualize pairplot của các biến được chọn với màu sắc được phân loại theo cụm
sns.pairplot(doc, hue='Cluster', palette='Dark2')
plt.title('Biểu đồ Pairplot của Các Biến Được Chọn với Cụm')
plt.show()
# Tính Silhouette Score
silhouette_avg = silhouette_score(features, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

# Tính Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(features, kmeans.labels_)
print(f"Davies-Bouldin Index: {davies_bouldin}")
print("sse_values:", sse_values)