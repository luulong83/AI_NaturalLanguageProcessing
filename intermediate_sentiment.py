import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from underthesea import word_tokenize
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return word_tokenize(text, format="text")

# Dữ liệu mẫu
data = {
    "text": [
        "Sản phẩm rất tuyệt vời và chất lượng",
        "Dịch vụ quá tệ, giao hàng chậm",
        "Hàng bình thường, không có gì đặc biệt",
        "Tôi rất hài lòng với sản phẩm này",
        "Máy bị lỗi, không sử dụng được",
        "Chất lượng ổn, giá cả hợp lý"
    ],
    "label": [0, 1, 2, 0, 1, 2]
}

# Tạo Dataset
df = pd.DataFrame(data)
df['text'] = df['text'].apply(preprocess_text)
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# Tải PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=3)

# Token hóa
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Hàm đánh giá
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./intermediate_results",
    # evaluation_strategy="epoch",
    eval_strategy="epoch",  # Thay evaluation_strategy thành eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
)

# Khởi tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Huấn luyện
trainer.train()

# Đánh giá
predictions = trainer.predict(tokenized_datasets["test"])
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# In kết quả
for text, pred, label in zip(dataset["test"]["text"], preds, labels):
    print(f"Văn bản: {text} | Dự đoán: {pred} | Thực tế: {label}")

# Vẽ confusion matrix
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()

# In số liệu
metrics = compute_metrics(predictions)
print(metrics)