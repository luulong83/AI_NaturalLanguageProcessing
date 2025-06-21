import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import random
import py_vncorenlp
import torch
import os

# Khởi tạo VnCoreNLP
py_vncorenlp.download_model(save_dir="C:/VnCoreNLP")
annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="C:/VnCoreNLP")

# Từ điển đồng nghĩa
synonyms_dict = {
    "đẹp": ["xinh", "lộng lẫy", "mỹ miều"],
    "tuyệt vời": ["xuất sắc", "hoàn hảo", "tuyệt diệu"],
    "tốt": ["tuyệt", "ok", "hài lòng"],
    "nhanh": ["mau", "lẹ", "tốc độ"],
    "ổn": ["tốt", "được", "hài lòng"],
    # Thêm các từ đồng nghĩa khác nếu cần
}

# Hàm thay thế từ đồng nghĩa
def synonym_replacement(comment):
    segmented_text = annotator.word_segment(comment)
    if segmented_text:
        words = segmented_text[0].split()
    else:
        words = comment.split()
    new_words = words.copy()
    for i, word in enumerate(words):
        if word in synonyms_dict and random.random() < 0.3:
            new_words[i] = random.choice(synonyms_dict[word])
    return " ".join(new_words)

# Hàm tăng cường dữ liệu
def augment_data(data):
    augmented_data = []
    for _, row in data.iterrows():
        comment = row["comment"]
        label = row["label"]
        rate = row["rate"]
        augmented_data.extend([
            {"comment": comment, "label": label, "rate": rate},
            {"comment": synonym_replacement(comment), "label": label, "rate": rate},
        ])
    return pd.DataFrame(augmented_data)

# Chạy chính
if __name__ == "__main__":
    try:
        # Lấy đường dẫn tuyệt đối đến thư mục gốc dự án
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        input_path = os.path.join(project_root, "data", "raw", "test_5k.csv")
        output_dir = os.path.join(project_root, "data", "processed")
        output_path = os.path.join(output_dir, "augmented_test_2k.csv")

        print(f"📄 Đang đọc file từ: {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

        data = pd.read_csv(input_path, usecols=["comment", "label", "rate"], on_bad_lines='skip')

        print("✅ Đọc file thành công, bắt đầu kiểm tra cấu trúc...")
        required_columns = {"comment", "label", "rate"}
        if not required_columns.issubset(data.columns):
            raise ValueError("❌ File CSV phải có đầy đủ các cột: comment, label, rate")

        print("🚀 Cấu trúc hợp lệ, bắt đầu tăng cường dữ liệu...")
        augmented_data = augment_data(data)

        os.makedirs(output_dir, exist_ok=True)
        augmented_data.to_csv(output_path, index=False)

        print(f"✅ Tăng cường dữ liệu hoàn tất. File đã lưu tại: {output_path}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
