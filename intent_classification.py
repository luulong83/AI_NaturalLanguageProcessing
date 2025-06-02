import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from underthesea import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Tiền xử lý văn bản tiếng Việt
def preprocess_text(texts):
    """Phân đoạn từ và chuẩn hóa văn bản tiếng Việt"""
    return [word_tokenize(text.lower(), format="text") for text in texts]

# 2. Tạo dữ liệu giả lập
data = {
    "text": [
        # 0: Đặt hàng
        "Tôi muốn mua một cái laptop mới.",
        "Cho tôi đặt một ly trà sữa size lớn.",
        "Mua giúp tôi 3 gói mì Hảo Hảo.",
        "Tôi cần đặt một chiếc vé xem phim.",
        "Đặt hàng dùm tôi 5kg gạo.",
        "Tôi muốn mua 2 chai dầu gội.",
        "Tôi muốn đặt bàn ăn tại nhà hàng.",
        "Đặt giúp tôi một bó hoa tươi.",
        "Tôi muốn mua iPhone 15 Pro Max.",
        "Tôi cần đặt một chuyến taxi đi sân bay.",

        # 1: Hỏi thông tin
        "Cho tôi hỏi giá của sản phẩm này.",
        "Thời gian giao hàng bao lâu vậy?",
        "Chính sách đổi trả như thế nào?",
        "Shop còn hàng không?",
        "Có khuyến mãi nào đang áp dụng không?",
        "Giờ làm việc của cửa hàng là khi nào?",
        "Phí vận chuyển là bao nhiêu?",
        "Cách thanh toán như thế nào?",
        "Sản phẩm này có bảo hành không?",
        "Có thể xem trước hàng không?",

        # 2: Yêu cầu hỗ trợ
        "Tôi không đăng nhập được vào tài khoản.",
        "Ứng dụng bị lỗi, tôi không mở được.",
        "Làm sao để cập nhật thông tin cá nhân?",
        "Tôi cần hỗ trợ đổi mật khẩu.",
        "Vui lòng giúp tôi kiểm tra đơn hàng.",
        "Tôi bị tính phí sai, xin kiểm tra lại.",
        "Không nhận được mã xác thực OTP.",
        "Trang web load rất chậm.",
        "Tôi không nhận được email xác nhận.",
        "Tôi cần hỗ trợ kỹ thuật.",

        # 3: Hủy đơn hàng
        "Tôi muốn hủy đơn hàng vừa đặt.",
        "Làm sao để hủy đơn?",
        "Tôi không cần sản phẩm này nữa.",
        "Hủy đơn hàng giúp tôi với.",
        "Tôi đã đặt nhầm, cần hủy đơn.",
        "Muốn đổi sản phẩm khác, hủy đơn cũ đi.",
        "Tôi muốn dừng đơn hàng đang xử lý.",
        "Tôi không muốn mua nữa.",
        "Vui lòng hủy đơn hàng này.",
        "Tôi đã mua ở chỗ khác rồi, xin hủy."
    ],
    "label": [
        *[0]*10,  # Đặt hàng
        *[1]*10,  # Hỏi thông tin
        *[2]*10,  # Yêu cầu hỗ trợ
        *[3]*10   # Hủy đơn hàng
    ]
}
data["text"] = preprocess_text(data["text"])

# 3. Tạo Dataset
dataset = Dataset.from_dict(data)
split_data = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]
eval_texts = split_data["test"]["text"]

# 4. Tải mô hình và tokenizer
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# 5. Mã hóa dữ liệu
def tokenize_function(examples):
    return tokenizer(
        [str(x) for x in examples["text"]],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# 6. Định nghĩa hàm tính toán số liệu
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return {
        "accuracy": accuracy,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist()
    }

# 7. Cấu hình huấn luyện
# 7. Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir='./results_intent_classification',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",  # Thay evaluation_strategy thành eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs_intent_classification',
    logging_steps=10,
    fp16=torch.cuda.is_available()
)


# 8. Khởi tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 9. Huấn luyện mô hình
trainer.train()

# 10. Dự đoán trên tập đánh giá
predictions = trainer.predict(eval_dataset)
pred_labels = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids

# 11. Chuyển nhãn số thành tên lớp
label_map = {0: "Đặt hàng", 1: "Hỏi thông tin", 2: "Yêu cầu hỗ trợ", 3: "Hủy đơn hàng"}

# 12. In kết quả phân loại
print("\nKết quả phân loại ý định:")
for text, pred, true in zip(eval_texts, pred_labels, true_labels):
    print(f"Văn bản: {text}")
    print(f"Dự đoán: {label_map[pred]}")
    print(f"Thực tế: {label_map[true]}")
    print("-" * 50)

# 13. Vẽ confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title("Confusion Matrix - Intent Classification")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 14. In chi tiết số liệu
metrics = compute_metrics(predictions)
print("\nSố liệu đánh giá chi tiết:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
for i, label in label_map.items():
    print(f"\nÝ định: {label}")
    print(f"Precision: {metrics['precision_per_class'][i]:.4f}")
    print(f"Recall: {metrics['recall_per_class'][i]:.4f}")
    print(f"F1-score: {metrics['f1_per_class'][i]:.4f}")

# 15. Lưu mô hình tốt nhất
trainer.save_model("./best_intent_model")
