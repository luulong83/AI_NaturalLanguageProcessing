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

### `data_augmentation.py` có nhiệm vụ tăng cường dữ liệu (data augmentation)
### `train_phobert_classifier.py`: Huấn luyện mô hình PhoBERT trên dữ liệu tăng cường (augmented_test_2k.csv), đánh giá accuracy, và lưu mô hình vào phobert_best.pt. Sử dụng utils.py để tải cấu hình và lưu mô hình.
### `inference.py`: Tải mô hình đã huấn luyện để dự đoán nhãn ("natural" hoặc "generated") cho văn bản mới hoặc toàn bộ file CSV. Lưu kết quả vào predictions.csv.
### `utils.py`: Cung cấp các hàm tiện ích như tải cấu hình, lưu/lấy mô hình, và tạo thư mục nếu cần. File config.json là tùy chọn (nếu muốn tùy chỉnh project_root).


# Thư viện:
`pip install pandas torch transformers py_vncorenlp`
python -m pip install pandas torch transformers py_vncorenlp
