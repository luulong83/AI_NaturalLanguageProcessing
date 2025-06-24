import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import random
import py_vncorenlp
import torch
import os

# Kiểm tra thiết bị
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Hàm back-translation

# def back_translation(comment):
#      model_name = "Helsinki-NLP/opus-mt-vi-en"
#      tokenizer_en = MarianTokenizer.from_pretrained(model_name)
#      model_en = MarianMTModel.from_pretrained(model_name).to(device)
#      model_name = "Helsinki-NLP/opus-mt-en-vi"
#      tokenizer_vi = MarianTokenizer.from_pretrained(model_name)
#      model_vi = MarianMTModel.from_pretrained(model_name).to(device)
     
#      inputs = tokenizer_en(comment, return_tensors="pt", truncation=True, max_length=128).to(device)
#      translated = model_en.generate(**inputs)
#      en_text = tokenizer_en.decode(translated[0], skip_special_tokens=True)
     
#      inputs = tokenizer_vi(en_text, return_tensors="pt", truncation=True, max_length=128).to(device)
#      translated = model_vi.generate(**inputs)
#      return tokenizer_vi.decode(translated[0], skip_special_tokens=True)

# Hàm tăng cường dữ liệu
def augment_data(data):
    augmented_data = []
    for _, row in data.iterrows():
        comment = row["comment"]
        label = row["label"]
        rate = row["rate"]
        augmented_data.extend(
            (
                {"comment": comment, "label": label, "rate": rate},
                {
                    "comment": synonym_replacement(comment),
                    "label": label,
                    "rate": rate,
                },
            )
        )
            # augmented_data.append({"comment": back_translation(comment), "label": label, "rate": rate})
    return pd.DataFrame(augmented_data)

# Áp dụng
if __name__ == "__main__":
    try:
        print("Danh sách file trong thư mục:", os.listdir("."))
        print("Bắt đầu đọc file...")
        data = pd.read_csv("C:/Users/HP/Documents/AI_NaturalLanguageProcessing/TaiLieu/test_5k_fixed.csv", usecols=["comment", "label", "rate"], on_bad_lines='skip')
        print("Đọc file thành công, bắt đầu kiểm tra cấu trúc...")
        if any(
            col not in data.columns for col in ["comment", "label", "rate"]
        ):
            raise ValueError("File CSV phải có các cột: comment, label, rate")
        print("Kiểm tra cấu trúc thành công, bắt đầu tăng cường dữ liệu...")
        augmented_data = augment_data(data)
        output_path = os.path.join(os.getcwd(), "augmented_test_2k.csv")
        print(f"Đường dẫn lưu file: {output_path}")
        augmented_data.to_csv(output_path, index=False)
        print("Tăng cường dữ liệu hoàn tất, đang lưu file...")
        print("Đã tạo file augmented_test_2k.csv thành công!")
    except Exception as e:
        print(f"Lỗi: {e}")