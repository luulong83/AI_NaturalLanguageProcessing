🧠 Bài toán: Chuẩn hóa văn bản viết tắt bằng học sâu
🎯 Mục tiêu:

Chuyển câu viết tắt như:
"t ko bik j lun"
→ "tôi không biết gì luôn"
bằng mô hình học từ dữ liệu, không cần từ điển thủ công.

🛠️ Giải pháp: Seq2Seq (Encoder-Decoder)

    Input: Văn bản viết tắt ("mik k h bik").

    Output: Văn bản chuẩn ("mình không hiểu biết").

    Mô hình: LSTM hoặc Transformer.

✅ Ví dụ thực thi (minh họa đơn giản với Hugging Face Transformers)
⚠️ Lưu ý: Đây là mô hình minh họa, cần tập dữ liệu đủ lớn để huấn luyện tốt.


✅ Format thống nhất cho bài mẫu NLP tiếng Việt
🧠 Tên bài:
    (Ví dụ: Chuẩn hóa văn bản viết tắt bằng học sâu)
🎯 Mục tiêu học tập:
    Mô tả ngắn về kỹ thuật NLP và kiến thức học máy áp dụng.
📁 Cấu trúc thư mục đề xuất:
    <ten_du_an>/
    ├── data/
    │   └── input_sample.txt
    ├── model/
    │   └── pretrained_or_finetuned_model/
    ├── utils/
    │   └── helpers.py
    ├── chuan_hoa_seq2seq.py   ← file chính
    └── README.md              ← hướng dẫn chạy
📄 File: chuan_hoa_seq2seq.py

🐍 Python version đề xuất:
    Python >= 3.8, <= 3.11
⚠️ Nên tránh Python 3.12 vì một số thư viện NLP chưa hỗ trợ tốt.
📦 Thư viện cần cài (requirements):
transformers==4.41.1
torch>=1.13.0
sentencepiece  # dùng cho tokenizer của T5
📄 Gợi ý requirements.txt:

📁 bai_tap_04_chuan_hoa/requirements.txt


Tạo lập môi trường ảo Virtualenv Environment
	# py -V:3.10 -m venv myenv
  	# cmd: .\myenv\Scripts\activate

Dự đoán (Inference)
Sau khi huấn luyện xong, chạy:  python infer.py

