import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# ==== CẤU HÌNH ĐƠN GIẢN ====
DATA_PATH = "data/processed/augmented_test_2k.csv"
MODEL_SAVE_PATH = "models/phobert_simple.pt"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== DATASET TÙY CHỈNH ====
class SimpleDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.sentences = dataframe["comment"].values
        self.labels = dataframe["label"].map({"POS": 0, "NEG": 1, "NEU": 2}).fillna(0).values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": label
        }


# ==== LOAD DỮ LIỆU ====
print(f"📄 Loading data from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy file: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
print(f"📊 Dữ liệu có {df.shape[0]} dòng, Nhãn: {df['label'].unique()}")

# ==== TOKENIZER & MODEL ====
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
model = model.to(DEVICE)

# ==== DATALOADER ====
dataset = SimpleDataset(df, tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== HUẤN LUYỆN ====
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

print("🚀 Bắt đầu huấn luyện...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(dataloader):.4f}")

# ==== ĐÁNH GIÁ ====
print("🔍 Bắt đầu đánh giá...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"🎯 Accuracy: {accuracy:.4f}")

# ==== LƯU MÔ HÌNH ====
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 Đã lưu mô hình tại: {MODEL_SAVE_PATH}")
