project_folder/
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   └── processed/            # Dữ liệu đã augment
├── models/
│   └── phobert_best.pt       # Model đã huấn luyện
├── src/
│   ├── data_augmentation.py  # Tăng cường dữ liệu
│   ├── train_phobert_classifier.py
│   ├── inference.py
│   └── utils.py
└── requirements.txt

Để tích hợp Topological Data Analysis (TDA) với kỹ thuật tăng cường dữ liệu trong dự án phân loại văn bản sử dụng PhoBERT như trong các file bạn cung cấp, chúng ta sẽ sử dụng các thư viện Ripser và Persim để phân tích cấu trúc topological của dữ liệu văn bản, từ đó tạo ra các đặc trưng bổ sung hoặc xác định các mẫu dữ liệu có thể được tăng cường. Dưới đây là hướng dẫn chi tiết, dễ hiểu, để bổ sung TDA vào quy trình tăng cường dữ liệu trong file data_augmentation.py.

## 1. Tổng quan về cách tích hợp TDA
TDA giúp phân tích cấu trúc hình học của dữ liệu thông qua các công cụ như persistent homology (được tính toán bởi Ripser) và khoảng cách giữa các biểu đồ bền vững (persistence diagrams, được tính bởi Persim). Trong ngữ cảnh này, chúng ta sẽ:

 - 1. `Chuyển đổi văn bản thành vector`: Sử dụng PhoBERT để tạo embedding cho các bình luận (comments) trong dữ liệu.
 - 2. `Áp dụng TDA`: Sử dụng Ripser để tính toán persistence diagrams từ các embedding này, giúp xác định các đặc trưng topological như "lỗ" (holes) hoặc "cụm" (clusters) trong dữ liệu.
    Phân tích với Persim: Tính khoảng cách giữa các persistence diagrams để xác định các bình luận tương tự hoặc khác biệt về mặt topological, từ đó chọn các bình luận cần tăng cường dữ liệu.
 -3. `Kết hợp với tăng cường dữ liệu`: Dựa trên phân tích TDA, ưu tiên tăng cường dữ liệu cho các bình luận thuộc các vùng topological ít đại diện (ví dụ: các cụm thưa thớt hoặc các điểm bất thường).
 -4. `Tích hợp vào pipeline`: Cập nhật file data_augmentation.py để thêm bước TDA trước khi thực hiện tăng cường dữ liệu.


## 2. Các bước triển khai
##  2.1. Cài đặt thư viện

Đảm bảo bạn đã cài đặt các thư viện cần thiết:
bash
`pip install torch transformers pandas ripser persim numpy scikit-learn`

## 2.2. Tạo embedding từ PhoBERT

Chúng ta sẽ sử dụng PhoBERT để tạo embedding cho các bình luận. Embedding này sẽ là đầu vào cho Ripser để tính toán persistence diagrams.
## 2.3. Sử dụng Ripser và Persim

`Ripser`: Tính toán persistence diagrams từ tập hợp các điểm (embedding).
`Persim`: Tính toán khoảng cách Wasserstein hoặc Bottleneck giữa các persistence diagrams để so sánh cấu trúc topological giữa các bình luận.

## 2.4. Tích hợp vào data_augmentation.py

Chúng ta sẽ cập nhật file data_augmentation.py để:
   -  Tạo embedding cho các bình luận.
   -  Phân tích topological để xác định các bình luận cần tăng cường.
   -  Áp dụng kỹ thuật thay thế từ đồng nghĩa (synonym replacement) cho các bình luận được chọn.


## 4. Giải thích mã nguồn
## 4.1. `Hàm get_phobert_embeddings`

    Chức năng: Tạo embedding cho từng bình luận bằng cách sử dụng PhoBERT, lấy vector của token [CLS] làm đại diện cho câu.
    Đầu vào: Danh sách các bình luận, tokenizer, và mô hình PhoBERT.
    Đầu ra: Ma trận embedding (numpy array).

## 4.2. `Hàm select_comments_for_augmentation`

    Chức năng: Sử dụng Ripser để tính persistence diagrams từ embedding, sau đó dùng Persim để tính khoảng cách topological. Các bình luận có khoảng cách lớn (ít đại diện trong không gian topological) sẽ được chọn để tăng cường.
    Đầu vào: Ma trận embedding.
    Đầu ra: Danh sách chỉ số của các bình luận cần tăng cường.

## 4.3. `Hàm augment_data`

    Chức năng: Kết hợp TDA với kỹ thuật thay thế từ đồng nghĩa. Chỉ các bình luận được chọn bởi TDA mới được tăng cường.
    Đầu vào: DataFrame chứa bình luận, tokenizer, và mô hình PhoBERT.
    Đầu ra: DataFrame chứa dữ liệu gốc và dữ liệu tăng cường.

## 4.4. `Hàm synonym_replacement`

    Hàm này giữ nguyên từ file gốc, thực hiện thay thế từ đồng nghĩa dựa trên từ điển synonyms_dict.


6. Lợi ích của việc tích hợp TDA

    `Xác định dữ liệu ít đại diện`: TDA giúp phát hiện các bình luận có cấu trúc topological khác biệt, từ đó ưu tiên tăng cường dữ liệu cho các vùng thưa thớt trong không gian dữ liệu.
    Cải thiện độ đa dạng: Dữ liệu tăng cường sẽ phong phú hơn, giúp mô hình học tốt hơn trên các mẫu hiếm.
    `Tăng độ chính xác`: PhoBERT sẽ được huấn luyện trên tập dữ liệu cân bằng hơn về mặt topological, cải thiện hiệu suất phân loại.

7. Lưu ý

    `Hiệu suất`: Tính toán persistence diagrams có thể tốn thời gian với tập dữ liệu lớn. Có thể tối ưu bằng cách giảm số chiều của embedding trước khi đưa vào Ripser (ví dụ: sử dụng PCA).
    Ngưỡng (threshold): Tham số threshold trong select_comments_for_augmentation cần được điều chỉnh dựa trên tập dữ liệu cụ thể.
    `Tài nguyên`: Đảm bảo máy có đủ RAM và GPU (nếu dùng CUDA) để xử lý embedding và tính toán TDA.

`*********************************************************************************************`
# 1. File train_phobert_classifier.py
`*********************************************************************************************`
## Phân tích:
   - File này chịu trách nhiệm huấn luyện mô hình PhoBERTClassifier và sử dụng dữ liệu từ file `augmented_test_2k.csv` (nay có thể là `augmented_test_2k_with_tda.csv` sau khi tích hợp TDA).
   -  Vì TDA chỉ ảnh hưởng đến dữ liệu đầu vào (tăng cường dữ liệu), mã nguồn trong `train_phobert_classifier.py` không cần thay đổi logic chính. Tuy nhiên, chúng ta có thể:
        - Cập nhật đường dẫn đến file dữ liệu tăng cường mới.
        - Thêm kiểm tra dữ liệu để đảm bảo dữ liệu từ TDA có định dạng phù hợp.
        - (Tùy chọn) Thêm khả năng đánh giá sự cải thiện từ dữ liệu tăng cường TDA bằng cách lưu thêm metric.

# Thay đổi đề xuất:

    - Cập nhật đường dẫn file dữ liệu trong phần __main__.
    - Thêm kiểm tra dữ liệu đầu vào để đảm bảo tính toàn vẹn.
    - (Tùy chọn) Lưu thêm thông tin về hiệu suất mô hình vào file log.
  
`*********************************************************************************************`
# 2. File inference.py
`*********************************************************************************************`
# Phân tích:

- File này thực hiện dự đoán trên văn bản đầu vào hoặc file CSV, sử dụng mô hình đã huấn luyện.
- TDA không ảnh hưởng trực tiếp đến quá trình suy luận, vì nó chỉ được áp dụng trong bước tăng cường dữ liệu (trước khi huấn luyện).
- Tuy nhiên, nếu bạn muốn dự đoán trên file dữ liệu tăng cường từ TDA, cần cập nhật đường dẫn đến file đầu vào trong phần __main__.

# Thay đổi đề xuất:

- Cập nhật đường dẫn đến file dữ liệu đầu vào (nếu cần dự đoán trên dữ liệu tăng cường).
- (Tùy chọn) Thêm thông báo để xác nhận dữ liệu đầu vào là từ pipeline TDA.



`*********************************************************************************************`
# 3. File utils.py
`*********************************************************************************************`
# Phân tích:

- File này chứa các hàm tiện ích như `load_config`, `save_model`, `load_model`, và `ensure_dir`.
- TDA không yêu cầu thay đổi trực tiếp trong file này, nhưng có thể bổ sung một hàm để lưu log hiệu suất (ví dụ: accuracy từ `train_phobert_classifier.py`) để theo dõi tác động của TDA.

# Thay đổi đề xuất:

    (Tùy chọn) Thêm hàm save_log để lưu thông tin về hiệu suất mô hình hoặc dữ liệu tăng cường.

# Mã nguồn cập nhật:

Bổ sung hàm save_log vào file `utils.py`: