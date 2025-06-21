import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
from utils import load_config, load_model

# Mô hình PhoBERT
class PhoBERTClassifier(nn.Module):
    def __init__(self, phobert_model):
        super().__init__()
        self.phobert = phobert_model
        self.dropout = nn.Dropout(0.1)
        # Loại bỏ lớp classifier tùy chỉnh, sử dụng classifier của phobert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        return outputs.logits  # Trả về logits trực tiếp

# Dự đoán
def predict(model, tokenizer, text, device, max_len=128):
    model.eval()
    inputs = tokenizer.encode_plus(
        text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).cpu().item()
    # Cập nhật ánh xạ cho POS, NEG, NEU
    label_map = {0: "POS", 1: "NEG", 2: "NEU"}
    return label_map.get(pred, "unknown")

if __name__ == "__main__":
    # Load cấu hình
    config = load_config()
    project_root = config["project_root"]
    model_path = os.path.join(project_root, "models", "phobert_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📥 Đang tải mô hình...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    # Sử dụng num_labels=3 để khớp với huấn luyện
    model = PhoBERTClassifier(AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3))
    model.load_state_dict(load_model(model_path))
    model = model.to(device)

    # Ví dụ dự đoán
    test_text = "Câu này có tự nhiên không?"
    result = predict(model, tokenizer, test_text, device)
    print(f"📝 Văn bản: {test_text}")
    print(f"🔍 Dự đoán: {result}")

    # Dự đoán từ file (tuỳ chọn)
    input_path = os.path.join(project_root, "data", "raw", "test_5k.csv")
    if os.path.exists(input_path):
        data = pd.read_csv(input_path, usecols=["comment"])
        predictions = [predict(model, tokenizer, text, device) for text in data["comment"]]
        data["prediction"] = predictions
        output_path = os.path.join(project_root, "data", "processed", "predictions.csv")
        data.to_csv(output_path, index=False)
        print(f"✅ Đã lưu kết quả dự đoán tại: {output_path}")