## Ứng dụng thuật toán học máy để phân tích giá nhà


### Sử dụng thuật toán K-means để phân cụm dữ liệu
Sử dụng các thư viện matplotlib, sklearn, pandas, time, seaborn 

Đọc và chọn dữ liệu

Chọn giá trị k (cụm) phù hợp bằng phương pháp Elbow và trực quan hóa kết quả

Vẽ đồ thị Elbow và thời gian (fit time) thực hiện K-means: Trực quan hóa để chọn K tối ưu

Tính các giá trị Silhouette Score và Davies bouldin Index để đánh giá chất lượng phân cụm

Dùng K-means để phân cụm dữ liệu: Áp dụng K-means với số cụm đã chọn

### Sử dụng thuật toán K-nearest neighbors Classifier để phân loại giá nhà
Sử dụng các thư viện pandas, sklearn, matplotlib, seaborn

Đọc dữ liệu từ file Excel

Tiền xử lí dữ liệu.
Tiến hành chia xột ‘Price’ thành 3 mức 0, 1, 2 tương ứng với “Rẻ, Trung bình, Đắt’.

Chọn đặc trưng và tiến hành chia dữ liệu thành tập huấn luyện và tập kiểm tra với 80% là tập huấn luyện và 20% là tập kiểm tra.

Chuẩn hóa dữ liệu
Chuẩn hóa dữ liệu để đảm bảo các đặc trưng có cùng thang đo, giúp mô hình học hiệu quả hơn

Xây dựng mô hình KNN
Sử dụng thuật toán K-Nearest Neighbors (KNN) để xây dựng mô hình phân loại, với số lượng láng giềng là 3.

Dự đoán và đánh giá mô hình
Dự đoán trên tập kiểm tra và đánh giá mô hình sử dụng độ chính xác Accuracy và ma trận nhầm lẫn.

Trực quan hóa kết quả
Trực quan hóa ma trận nhầm lẫn bằng heatmap để thấy cách mô hình phân loại giá nhà vào các mức giá. 

### Sử dụng thuật toán K-nearest neighbors Regressor để phân loại giá nhà
Sử dụng các thư viện pandas, sklearn, matplotlib, seaborn

Đọc dữ liệu từ file Excel

Chọn đặc trưng và biến mục tiêu

Chia dữ liệu thành tập huấn luyện và kiểm thử

Chuẩn hóa dữ liệu để đảm bảo các đặc trưng có cùng tỷ lệ

Xây dựng mô hình hồi quy K-nearest Neighbors Regressor

Dự đoán trên tập kiểm thử và đánh giá mô hình

Trực quan hóa kết quả
Vẽ đồ thị phân tán giữa giá trị thực tế và giá trị dự đoán, và thêm đường hồi quy dự kiến (đường chéo đỏ). Điều này giúp hiển thị mức độ chính xác của mô hình và mối quan hệ giữa dự đoán và giá trị thực tế. 

### Sử dụng hồi quy tuyến tính – Linear Regression để dự đoán giá nhà
Sử dụng các thư viện pandas, sklearn, matplotlib, seaborn

Đọc dữ liệu từ file Excel

Chia dữ liệu thành tập huấn luyện và kiểm tra

Chuẩn hóa dữ liệu
Sử dụng StandarScaler để chuẩn hóa dữ liệu để có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1.

Xây dựng mô hình Linear Regression
Sử dụng thư viện scikit-learn để xây dựng mô hình Linear Regression trên tập huấn luyện

Dự đoán và đánh giá mô hình
Dự đoán giá nhà trên tập kiểm thử và tính toán MSE và R-squred để đánh giá hiệu suất của mô hình.

Tính Residuals bằng cách lấy hiệu giữa giá trị thực tế và giá trị dự đoán.

Trực quan hóa kết quả