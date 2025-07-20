import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report
from utils import load_config, load_model  # Giả định utils có các hàm này

# Hàm dự đoán
# Ghi chú: Dự đoán nhãn cho một văn bản, xử lý lỗi văn bản không hợp lệ
def predict(model, tokenizer, text, device, max_len=128):
    if not text or not isinstance(text, str):
        return "unknown"
    
    model.eval()
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
            pred = torch.argmax(outputs.logits, dim=1).cpu().item()
        label_map = {0: "POS", 1: "NEG", 2: "NEU"}
        return label_map.get(pred, "unknown")
    except Exception as e:
        print(f"⚠ Lỗi khi dự đoán văn bản: {text}. Lỗi: {e}")
        return "unknown"

# Hàm dự đoán theo batch để cải thiện hiệu suất
# Ghi chú: Xử lý nhiều văn bản cùng lúc để tăng tốc
def predict_batch(model, tokenizer, texts, device, max_len=128, batch_size=32):
    model.eval()
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            [text for text in batch_texts if isinstance(text, str) and text.strip()],
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        
        label_map = {0: "POS", 1: "NEG", 2: "NEU"}
        predictions.extend([label_map.get(pred, "unknown") for pred in preds])
        
        # Thêm nhãn "unknown" cho các văn bản không hợp lệ
        predictions.extend(["unknown"] * (len(batch_texts) - len(preds)))
    
    return predictions

if __name__ == "__main__":
    try:
        # Load cấu hình
        config = load_config()
        project_root = config["project_root"]
        model_path = os.path.join(project_root, "models", "phobert_best.pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥 Sử dụng thiết bị: {device}")

        # Load tokenizer và mô hình
        print("📥 Đang tải mô hình...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
        model.load_state_dict(load_model(model_path))
        model = model.to(device)

        # Ví dụ dự đoán đơn lẻ
        test_text = "Câu này có tự nhiên không?"
        result = predict(model, tokenizer, test_text, device)
        print(f"📝 Văn bản: {test_text}")
        print(f"🔍 Dự đoán: {result}")

        # Dự đoán từ file tăng cường
        input_path = os.path.join(project_root, "data", "processed", "augmented_test_2k_with_tda.csv")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

        print(f"📄 Đang đọc file từ: {input_path}")
        data = pd.read_csv(input_path, usecols=["comment", "label"])
        data = data.dropna(subset=["comment"])  # Loại bỏ hàng có comment NaN
        data = data[data["comment"].str.strip() != ""]  # Loại bỏ bình luận rỗng

        # Dự đoán theo batch
        print("🚀 Đang dự đoán trên dữ liệu tăng cường...")
        predictions = predict_batch(model, tokenizer, data["comment"].tolist(), device)

        # Lưu kết quả
        data["prediction"] = predictions
        output_path = os.path.join(project_root, "data", "processed", "predictions_augmented.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"✅ Đã lưu kết quả dự đoán tại: {output_path}")

        # Đánh giá nếu có nhãn thực tế
        if "label" in data.columns:
            print("📊 Đánh giá hiệu quả mô hình:")
            print(f"Số bình luận được dự đoán: {len(predictions)}")
            print("Phân bố nhãn dự đoán:")
            print(pd.Series(predictions).value_counts())
            print("Phân bố nhãn thực tế:")
            print(data["label"].value_counts())
            
            # Tính độ chính xác và báo cáo chi tiết
            accuracy = accuracy_score(data["label"], predictions)
            print(f"Độ chính xác: {accuracy:.4f}")
            print("Báo cáo chi tiết:")
            print(classification_report(data["label"], predictions, target_names=["POS", "NEG", "NEU"]))

    except Exception as e:
        print(f"❌ Lỗi: {e}")