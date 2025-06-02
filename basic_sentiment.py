# Cấp Cơ Bản: Phân Loại Cảm Xúc Đơn Giản
# Mục tiêu: Làm quen với tiền xử lý và sử dụng PhoBERT để phân loại cảm xúc cơ bản.
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from underthesea import word_tokenize
import re

# Tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return word_tokenize(text, format="text")

# Dữ liệu mẫu
data = {
    "text": [
        "Sản phẩm này rất tốt",
        "Dịch vụ quá tệ",
        "Hàng bình thường thôi"
    ],
    "label": [0, 1, 2]  # 0: Tích cực, 1: Tiêu cực, 2: Trung tính
}

# Tạo Dataset
df = pd.DataFrame(data)
df['text'] = df['text'].apply(preprocess_text)
dataset = Dataset.from_pandas(df)

# Tải PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=3)

# Token hóa
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./basic_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_strategy="epoch",
)

# Khởi tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Huấn luyện
trainer.train()

# Dự đoán
from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = "Sản phẩm này không tuyệt vời"
result = classifier(preprocess_text(text))
print(result)