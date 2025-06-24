import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset tùy chỉnh
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.sentences = data["comment"].values  # Sử dụng cột "comment" thay vì "sentence"
        self.labels = data["label"].map({"natural": 0, "generated": 1}).values
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode_plus(
            sentence, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": label
        }

# Mô hình PhoBERT
class PhoBERTClassifier(nn.Module):
    def __init__(self, phobert_model):
        super().__init__()
        self.phobert = phobert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)  # 768 từ PhoBERT CLS, 2 lớp
    
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.logits  # (batch, 768)
        return self.classifier(self.dropout(cls_output))

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

# Áp dụng
if __name__ == "__main__":
    try:
        print("Danh sách file trong thư mục:", os.listdir("."))
        print("Bắt đầu đọc file...")
        data = pd.read_csv("test_5k_fixed.csv")  # Sử dụng file hiện có
        
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)
        
        dataset = TextDataset(data, tokenizer, max_len=128)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model = PhoBERTClassifier(phobert)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        
        print("Bắt đầu huấn luyện...")
        train_model(model, dataloader, optimizer, device)
        print("Huấn luyện hoàn tất, bắt đầu đánh giá...")
        accuracy = evaluate_model(model, dataloader, device)
        
        print("Lưu mô hình...")
        torch.save(model.state_dict(), "phobert_model.pth")
        print("Đã lưu mô hình thành công!")
    except Exception as e:
        print(f"Lỗi: {e}")