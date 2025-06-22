import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, AutoModel, AutoTokenizer
import random
import py_vncorenlp
import torch
import os
import numpy as np
from ripser import ripser
from persim import plot_diagrams, wasserstein
from sklearn.preprocessing import StandardScaler

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
}

# Hàm lấy embedding từ PhoBERT
def get_phobert_embeddings(texts, tokenizer, model, device, max_len=128):
    model.eval()
    embeddings = []
    for text in texts:
        inputs = tokenizer.encode_plus(
            text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Lấy [CLS] token
        embeddings.append(embedding[0])
    return np.array(embeddings)

# Hàm tính persistence diagrams và chọn bình luận cần tăng cường
def select_comments_for_augmentation(embeddings, threshold=0.5):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Tính persistence diagrams với Ripser
    diagrams = ripser(embeddings_scaled, maxdim=1)['dgms']
    
    # Tính khoảng cách Wasserstein giữa các điểm
    distances = []
    for i in range(len(embeddings)):
        dist = wasserstein(diagrams[0], diagrams[0], i, i)  # So sánh với chính nó (placeholder)
        distances.append(dist)
    
    # Chọn các bình luận có khoảng cách topological lớn (ít đại diện)
    selected_indices = [i for i, dist in enumerate(distances) if dist > threshold]
    return selected_indices

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

# Hàm tăng cường dữ liệu với TDA
def augment_data(data, tokenizer, model, device):
    augmented_data = []
    comments = data["comment"].values
    
    print("🚀 Tạo embedding từ PhoBERT...")
    embeddings = get_phobert_embeddings(comments, tokenizer, model, device)
    
    print("🔍 Phân tích topological với Ripser...")
    selected_indices = select_comments_for_augmentation(embeddings)
    
    print(f"✅ Đã chọn {len(selected_indices)} bình luận để tăng cường.")
    
    for idx, row in data.iterrows():
        comment = row["comment"]
        label = row["label"]
        rate = row["rate"]
        augmented_data.append({"comment": comment, "label": label, "rate": rate})
        
        # Chỉ tăng cường dữ liệu cho các bình luận được chọn bởi TDA
        if idx in selected_indices:
            augmented_comment = synonym_replacement(comment)
            augmented_data.append({"comment": augmented_comment, "label": label, "rate": rate})
    
    return pd.DataFrame(augmented_data)

# Chạy chính
if __name__ == "__main__":
    try:
        # Load cấu hình và tokenizer
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        input_path = os.path.join(project_root, "data", "raw", "test_5k.csv")
        output_dir = os.path.join(project_root, "data", "processed")
        output_path = os.path.join(output_dir, "augmented_test_2k_with_tda.csv")

        print(f"📄 Đang đọc file từ: {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

        data = pd.read_csv(input_path, usecols=["comment", "label", "rate"], on_bad_lines='skip')

        print("✅ Đọc file thành công, bắt đầu kiểm tra cấu trúc...")
        required_columns = {"comment", "label", "rate"}
        if not required_columns.issubset(data.columns):
            raise ValueError("❌ File CSV phải có đầy đủ các cột: comment, label, rate")

        # Load PhoBERT để tạo embedding
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModel.from_pretrained("vinai/phobert-base").to(device)

        print("🚀 Bắt đầu tăng cường dữ liệu với TDA...")
        augmented_data = augment_data(data, tokenizer, model, device)

        os.makedirs(output_dir, exist_ok=True)
        augmented_data.to_csv(output_path, index=False)

        print(f"✅ Tăng cường dữ liệu hoàn tất. File đã lưu tại: {output_path}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")