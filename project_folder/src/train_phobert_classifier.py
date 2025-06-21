import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from utils import load_config, save_model

# Dataset tùy chỉnh
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.sentences = data["comment"].values
        self.labels = data["label"].map({"POS": 0, "NEG": 1, "NEU": 2}).fillna(0).values
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode_plus(
            sentence, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": label
        }

# Mô hình PhoBERT
class PhoBERTClassifier(nn.Module):
    def __init__(self, phobert_model, num_labels=3):
        super().__init__()
        self.phobert = phobert_model
        self.dropout = nn.Dropout(0.1)
        # Không cần thêm classifier, sử dụng classifier của phobert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        return outputs.logits

# Huấn luyện
def train_model(model, dataloader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Đánh giá
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
    # Load cấu hình
    config = load_config()
    project_root = config["project_root"]
    data_path = os.path.join(project_root, "data", "processed", "augmented_test_2k.csv")
    model_path = os.path.join(project_root, "models", "phobert_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📄 Đang đọc file từ: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file: {data_path}")

    data = pd.read_csv(data_path, on_bad_lines='skip')  # Bỏ qua dòng lỗi
    print(f"📊 Dữ liệu: {data.shape}, Cột label: {data['label'].unique()}")  # Debug
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)

    dataset = TextDataset(data, tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = PhoBERTClassifier(phobert, num_labels=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    print("🚀 Bắt đầu huấn luyện...")
    train_model(model, dataloader, optimizer, device)
    print("✅ Huấn luyện hoàn tất, bắt đầu đánh giá...")
    accuracy = evaluate_model(model, dataloader, device)

    print("💾 Lưu mô hình...")
    save_model(model, model_path)
    print(f"✅ Đã lưu mô hình tại: {model_path}")