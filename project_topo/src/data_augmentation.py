import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import random
import py_vncorenlp
import torch
from ripser import ripser
from persim import wasserstein
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from persim import plot_diagrams

# Khởi tạo VnCoreNLP để phân đoạn từ tiếng Việt
# Ghi chú: Đường dẫn save_dir phải tồn tại và có quyền ghi
py_vncorenlp.download_model(save_dir="C:/VnCoreNLP")
annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="C:/VnCoreNLP")

# Từ điển đồng nghĩa cho việc thay thế từ
synonyms_dict = {
    "đẹp": ["xinh", "lộng lẫy", "mỹ miều"],
    "tuyệt vời": ["xuất sắc", "hoàn hảo", "tuyệt diệu"],
    "tốt": ["tuyệt", "ok", "hài lòng"],
    "nhanh": ["mau", "lẹ", "tốc độ"],
    "ổn": ["tốt", "được", "hài lòng"],
}

# Hàm lấy embedding từ PhoBERT
# Ghi chú: Hàm này tạo biểu diễn vector cho mỗi bình luận, xử lý lỗi và các trường hợp rỗng
def get_phobert_embeddings(texts, tokenizer, model, device, max_len=128):
    model.eval()
    embeddings = []
    for text in texts:
        # Kiểm tra văn bản rỗng hoặc không phải chuỗi
        if not text or not isinstance(text, str):
            embeddings.append(np.zeros(model.config.hidden_size))
            continue
        try:
            inputs = tokenizer.encode_plus(
                text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Lấy [CLS] token
            embeddings.append(embedding[0])
        except Exception as e:
            print(f"⚠ Lỗi khi xử lý văn bản: {text}. Lỗi: {e}")
            embeddings.append(np.zeros(model.config.hidden_size))
    return np.array(embeddings)

# Hàm chọn các bình luận cần tăng cường dựa trên TDA
# Ghi chú: Sử dụng persistent homology và Wasserstein distance để tìm các điểm bất thường
def select_comments_for_augmentation(embeddings, threshold=0.5, sample_size=1000):
    # Chuẩn hóa dữ liệu để đảm bảo TDA hoạt động đúng
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Lấy mẫu ngẫu nhiên nếu tập dữ liệu quá lớn để giảm chi phí tính toán
    if len(embeddings_scaled) > sample_size:
        indices = np.random.choice(len(embeddings_scaled), sample_size, replace=False)
        embeddings_sampled = embeddings_scaled[indices]
    else:
        embeddings_sampled = embeddings_scaled
        indices = np.arange(len(embeddings_scaled))
    
    # Tính persistence diagrams với Ripser
    print("🔍 Tính persistence diagrams với Ripser...")
    diagrams = ripser(embeddings_sampled, maxdim=1)['dgms']
    
    # Trực quan hóa persistence diagrams
    print("📊 Vẽ persistence diagrams...")
    plot_diagrams(diagrams, show=False)
    plt.savefig("persistence_diagrams.png")
    plt.close()
    
    # Tính Wasserstein distance giữa diagram của toàn bộ dữ liệu và từng tập con
    selected_indices = []
    full_diagram = diagrams[0]  # Diagram của toàn bộ dữ liệu
    for i, idx in enumerate(indices):
        # Tạo tập con không chứa điểm i
        subset_indices = [j for j in range(len(embeddings_sampled)) if j != i]
        subset_embeddings = embeddings_sampled[subset_indices]
        subset_diagram = ripser(subset_embeddings, maxdim=1)['dgms'][0]
        dist = wasserstein(full_diagram, subset_diagram)
        if dist > threshold:
            selected_indices.append(idx)  # Lưu chỉ số gốc của bình luận
    return selected_indices

# Hàm thay thế từ đồng nghĩa
# Ghi chú: Sử dụng VnCoreNLP để phân đoạn từ và thay thế ngẫu nhiên các từ đồng nghĩa
def synonym_replacement(comment):
    try:
        segmented_text = annotator.word_segment(comment)
        if segmented_text:
            words = segmented_text[0].split()
        else:
            words = comment.split()
    except Exception as e:
        print(f"⚠ Lỗi phân đoạn từ cho bình luận: {comment}. Lỗi: {e}")
        words = comment.split()
    
    new_words = words.copy()
    for i, word in enumerate(words):
        if word in synonyms_dict and random.random() < 0.3:
            new_words[i] = random.choice(synonyms_dict[word])
    return " ".join(new_words)

# Hàm tăng cường dữ liệu với TDA
# Ghi chú: Kết hợp embedding PhoBERT và TDA để chọn các bình luận cần tăng cường
def augment_data(data, tokenizer, model, device):
    augmented_data = []
    comments = data["comment"].values
    
    # Tạo embedding từ PhoBERT
    print("🚀 Tạo embedding từ PhoBERT...")
    embeddings = get_phobert_embeddings(comments, tokenizer, model, device)
    
    # Chọn các bình luận để tăng cường dựa trên TDA
    print("🔍 Phân tích topological với Ripser...")
    selected_indices = select_comments_for_augmentation(embeddings)
    
    print(f"✅ Đã chọn {len(selected_indices)} bình luận để tăng cường.")
    
    # Tạo tập dữ liệu tăng cường
    for idx, row in data.iterrows():
        comment = row["comment"]
        label = row["label"]
        rate = row["rate"]
        augmented_data.append({"comment": comment, "label": label, "rate": rate})
        
        # Tăng cường dữ liệu cho các bình luận được chọn
        if idx in selected_indices:
            augmented_comment = synonym_replacement(comment)
            augmented_data.append({"comment": augmented_comment, "label": label, "rate": rate})
    
    return pd.DataFrame(augmented_data)

# Chạy chính
if __name__ == "__main__":
    try:
        # Thiết lập đường dẫn file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        input_path = os.path.join(project_root, "data", "raw", "test_5k.csv")
        output_dir = os.path.join(project_root, "data", "processed")
        output_path = os.path.join(output_dir, "augmented_test_2k_with_tda.csv")

        # Đọc dữ liệu
        print(f"📄 Đang đọc file từ: {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

        data = pd.read_csv(input_path, usecols=["comment", "label", "rate"], on_bad_lines='skip')
        
        # Làm sạch dữ liệu
        print("✅ Đọc file thành công, bắt đầu làm sạch dữ liệu...")
        data = data.dropna(subset=["comment", "label", "rate"])  # Loại bỏ hàng NaN
        data = data[data["comment"].str.strip() != ""]  # Loại bỏ bình luận rỗng
        
        # Kiểm tra cấu trúc dữ liệu
        required_columns = {"comment", "label", "rate"}
        if not required_columns.issubset(data.columns):
            raise ValueError("❌ File CSV phải có đầy đủ các cột: comment, label, rate")

        # Load PhoBERT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥 Sử dụng thiết bị: {device}")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModel.from_pretrained("vinai/phobert-base").to(device)

        # Tăng cường dữ liệu
        print("🚀 Bắt đầu tăng cường dữ liệu với TDA...")
        augmented_data = augment_data(data, tokenizer, model, device)

        # Lưu kết quả
        os.makedirs(output_dir, exist_ok=True)
        augmented_data.to_csv(output_path, index=False)

        # Đánh giá kết quả tăng cường
        print(f"✅ Tăng cường dữ liệu hoàn tất. File đã lưu tại: {output_path}")
        print(f"Số bình luận trước tăng cường: {len(data)}")
        print(f"Số bình luận sau tăng cường: {len(augmented_data)}")
        print("📊 Phân bố nhãn trước tăng cường:")
        print(data["label"].value_counts())
        print("📊 Phân bố nhãn sau tăng cường:")
        print(augmented_data["label"].value_counts())

    except Exception as e:
        print(f"❌ Lỗi: {e}")