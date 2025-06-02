"""Cấp Nâng Cao: Tối Ưu Hóa và Triển Khai API

Mục tiêu: Tối ưu hóa mô hình với Optuna và triển khai API dự đoán bằng FastAPI."""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from underthesea import word_tokenize
import re
import optuna
from fastapi import FastAPI
from pydantic import BaseModel

# Tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return word_tokenize(text, format="text")

# Dữ liệu mẫu
data = {
    "text": [
        "Sản phẩm này rất tuyệt vời và chất lượng",
        "Dịch vụ quá tệ, giao hàng chậm",
        "Hàng bình thường, không có gì đặc biệt",
        "Tôi rất hài lòng với sản phẩm này",
        "Máy bị lỗi, không sử dụng được",
        "Chất lượng ổn, giá cả hợp lý",
        "Giao hàng nhanh, rất ưng ý",
        "Sản phẩm kém, nhanh hỏng",
        "Dịch vụ tốt nhưng giá hơi cao",
        "Không có gì để phàn nàn"
    ],
    "label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
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

# Tối ưu hóa hyperparameter với Optuna
def objective(trial):
    training_args = TrainingArguments(
        output_dir="./advanced_results",
        evaluation_strategy="epoch",
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=trial.suggest_float("weight_decay", 0.01, 0.1),
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.evaluate()["eval_accuracy"]

# Chạy Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Huấn luyện với tham số tốt nhất
best_params = study.best_params
training_args = TrainingArguments(
    output_dir="./advanced_results",
    evaluation_strategy="epoch",
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=best_params["weight_decay"],
    fp16=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("./phobert_advanced_sentiment")

# Tạo API với FastAPI
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    classifier = pipeline("text-classification", model="./phobert_advanced_sentiment", tokenizer=tokenizer)
    processed_text = preprocess_text(input.text)
    result = classifier(processed_text)
    return {"sentiment": result[0]["label"], "confidence": result[0]["score"]}