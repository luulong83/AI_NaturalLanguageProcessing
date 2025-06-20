import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# Dataset tùy chỉnh
class TextTopoDataset(Dataset):
    def __init__(self, data, topo_features, tokenizer, max_len):
        self.sentences = data["sentence"].values
        self.labels = data["label"].map({"natural": 0, "generated": 1}).values
        self.topo_features = topo_features  # (layers, heads, thresholds, samples, features)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode_plus(
            sentence, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        topo = torch.tensor(self.topo_features[:, :, :, idx, :].flatten(), dtype=torch.float32)  # (layers * heads * thresholds * features)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "topo_features": topo,
            "label": label
        }

# Mô hình kết hợp
class PhoBERTTopoClassifier(nn.Module):
    def __init__(self, phobert_model, topo_dim, hidden_dim=128):
        super().__init__()
        self.phobert = phobert_model
        self.topo_fc = nn.Linear(topo_dim, hidden_dim)
        self.classifier = nn.Linear(768 + hidden_dim, 2)  # 768 từ PhoBERT CLS, 2 lớp
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, topo_features):
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.logits  # (batch, 768)
        topo_output = self.relu(self.topo_fc(topo_features))  # (batch, hidden_dim)
        combined = torch.cat((cls_output, topo_output), dim=1)  # (batch, 768 + hidden_dim)
        return self.classifier(self.dropout(combined))

# Huấn luyện
def train_model(model, dataloader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            topo_features = batch["topo_features"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, topo_features)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Áp dụng
data = pd.read_csv("small_gpt_web/augmented_test_15k.csv")
topo_features = np.load("small_gpt_web/topo_features.npy")  # (12, 12, 6, samples, 3)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)

dataset = TextTopoDataset(data, topo_features, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = PhoBERTTopoClassifier(phobert, topo_dim=12*12*6*3)  # topo_features flattened
model = model.to("cuda:1")
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

train_model(model, dataloader, optimizer, "cuda:1")