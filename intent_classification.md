# Phân Loại Ý Định Tiếng Việt Bằng PhoBERT

## Mục Tiêu

* Xây dựng mô hình PhoBERT cho bài toán **phân loại ý định** văn bản tiếng Việt.
* Thực hành tiền xử lý văn bản tiếng Việt.
* Đánh giá chi tiết hiệu suất mô hình.
* Triển khai mô hình và tối ưu hóa huấn luyện.

## Mô Tả Bài Toán

* Phân loại văn bản tiếng Việt vào 4 ý định:

  * Đặt hàng (0)
  * Hỏi thông tin (1)
  * Yêu cầu hỗ trợ (2)
  * Hủy đơn hàng (3)

## Các Bước Thực Hiện

### 1. Tiền Xử Lý Dữ Liệu

* Sử dụng `underthesea` để **phân đoạn từ**.
* Chuyển văn bản về chữ thường.

### 2. Tạo Tập Dữ Liệu

* Tạo danh sách câu văn bản có gán nhãn.
* Sử dụng `datasets.Dataset` chia train/test theo tỉ lệ 80/20.

### 3. Tải PhoBERT

* Model: `vinai/phobert-base-v2`
* Tokenizer tùy chọn từ Hugging Face.

### 4. Token hóa dữ liệu

* Dùng tokenizer của PhoBERT để chuẩn bị dữ liệu cho huấn luyện.

### 5. Tính Toán Số Liệu

* Đánh giá: accuracy, precision, recall, F1-score theo từng lớp.

### 6. Cài Đặt Huấn Luyện

* Dùng `Trainer` với `TrainingArguments`:

  * Epoch = 10
  * Batch size = 4
  * Early stopping
  * GPU (FP16) nếu có

### 7. Huấn Luyện & Lưu Mô Hình

* Gồi `trainer.train()` và lưu mô hình tốt nhất.

### 8. Đánh Giá Kết Quả

* In dự đoán, nhãn thực, văn bản.
* Vẽ confusion matrix.
* In precision/recall/F1 chi tiết theo lớp.

### 9. Triển Khai Dự Đoán

* Dùng pipeline để dự đoán văn bản mới.

## Các Hướng Nghiên Cứu Sâu

### Tiền Xử

* Thêm stopwords, chuẩn hóa dấu câu.
* So sánh `underthesea` và `vncorenlp`.

### Dữ Liệu Không Cân Bằng

* Sử dụng class weights hoặc oversampling.

### Tối Ưu Hóa

* Dùng `Optuna` tìm hyperparameters.
* Thữa gradient\_accumulation\_steps.

### Trực Quan Hóa

* Vẽ đồ thị loss theo epoch.
* Vẽ confusion matrix.

### Triển Khai

* Dùng FastAPI tạo API dự đoán.

## Cài Đặt & Chạy

```bash
pip install torch transformers datasets scikit-learn underthesea seaborn matplotlib
python intent_classification.py
```

## Bài Tập Mở Rộng

* Thu thập dữ liệu thực tế
* So sánh nhiều mô hình NLP (XLM-RoBERTa, mBERT)
* Triển khai chatbot thực tế

## Tài Liệu Tham Khảo

* [PhoBERT Paper](https://arxiv.org/pdf/2003.00744)
* [PhoBERT GitHub](https://github.com/VinAIResearch/PhoBERT)
* [Transformers](https://huggingface.co/docs/transformers)
* [Intent Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)

## Kết Luận

Bài tập này giúc luyện tập NLP tiếng Việt với bài toán phân loại ý định, có tính ứng dụng cao trong chatbot/trợ lý ảo. Từ khâu tiền xử, huấn luyện, đánh giá, lưu và triển khai, người học có thể thực hành toàn diện quy trình NLP thực tế.
