"""
Bài Mẫu Cải Tiến (Cấp Cơ Bản)
Dưới đây là bài mẫu cải tiến cho phân loại cảm xúc đơn giản, với dữ liệu lớn hơn, chia train/test, và ánh xạ nhãn rõ ràng.
Lưu Ý
    Dữ liệu: Bài mẫu sử dụng 10 câu để cải thiện khả năng học của mô hình. Bạn có thể thêm dữ liệu thực tế từ bình luận mạng xã hội hoặc đánh giá sản phẩm.
    Cải tiến tiếp theo: Nếu muốn nâng cao, thử bài mẫu trung bình hoặc nâng cao từ câu trả lời trước, bao gồm trực quan hóa và tối ưu hóa hyperparameter.
    Sử dụng GPU: Nếu có GPU, thêm fp16=True trong TrainingArguments để tăng tốc.
"""
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
        "Sản phẩm này rất tốt và đáng mua",
        "Dịch vụ quá tệ giao hàng chậm",
        "Hàng bình thường không đặc biệt",
        "Tôi rất hài lòng với chất lượng",
        "Sản phẩm lỗi không dùng được",
        "Giá cả hợp lý chất lượng ổn",
        "Giao hàng nhanh rất tuyệt vời",
        "Không hài lòng với dịch vụ này",
        "Sản phẩm ok nhưng giá hơi cao",
        "Chất lượng sản phẩm rất tệ"
    ],
    "label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]  # 0: Tích cực, 1: Tiêu cực, 2: Trung tính
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

# Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./improved_basic_results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # evaluation_strategy="epoch",
    eval_strategy="epoch",  # Thay evaluation_strategy thành eval_strategy
    logging_strategy="epoch",
    learning_rate=2e-5,
)

# Hàm đánh giá
def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

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
print("Đánh giá trên tập test:", compute_metrics(predictions))

# Dự đoán văn bản mới
from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = "Sản phẩm này không tuyệt vời"
result = classifier(preprocess_text(text))
print(f"Dự đoán: {result}")