from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, pipeline
import pandas as pd
from datasets import Dataset

# --------------------
# 1. Load dữ liệu
# --------------------
train_df = pd.read_csv("data/train.tsv", sep="\t")
val_df = pd.read_csv("data/val.tsv", sep="\t")

# --------------------
# 2. Load tokenizer và model
# --------------------
try:
    tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")
    model = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base")
except Exception as e:
    print(f"❌ Lỗi tải mô hình hoặc tokenizer: {e}")
    exit()

# --------------------
# 3. Hàm tiền xử lý
# --------------------
def preprocess(examples):
    inputs = ["chuan_hoa: " + x for x in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=64, padding="max_length", truncation=True)
    labels = tokenizer(examples["output"], max_length=64, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

try:
    train_dataset = Dataset.from_pandas(train_df).map(preprocess, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(preprocess, batched=True)
    print("✅ Mẫu dữ liệu sau tiền xử lý:", train_dataset[0])
except Exception as e:
    print(f"❌ Lỗi trong tiền xử lý: {e}")
    exit()

# --------------------
# 4. Cấu hình huấn luyện
# --------------------
training_args = TrainingArguments(
    output_dir="./vit5_chuanhoa_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,  # Tăng để học tốt hơn
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to="none",
    fp16=False,
    gradient_accumulation_steps=1,
)

# --------------------
# 5. Tạo Trainer và huấn luyện
# --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

try:
    trainer.train()
except Exception as e:
    print(f"❌ Lỗi trong quá trình huấn luyện: {e}")
    exit()

# --------------------
# 6. Lưu mô hình và tokenizer
# --------------------
try:
    trainer.save_model("./vit5_chuanhoa_model")
    tokenizer.save_pretrained("./vit5_chuanhoa_model")
    print("✅ Mô hình và tokenizer đã được lưu vào ./vit5_chuanhoa_model")
except Exception as e:
    print(f"❌ Lỗi khi lưu mô hình: {e}")

# --------------------
# 7. Kiểm tra tokenizer
# --------------------
print("🔍 Kiểm tra tokenizer:")
test_input = "chuan_hoa: mik k hieu j het"
encoded = tokenizer(test_input, return_tensors="pt")
decoded = tokenizer.decode(encoded.input_ids[0], skip_special_tokens=True)
print(f"Input: {test_input}")
print(f"Encoded input_ids: {encoded.input_ids}")
print(f"Decoded back: {decoded}")

# --------------------
# 8. Kiểm tra mô hình
# --------------------
print("\n🔍 Kiểm tra mô hình:")
model.eval()

# Kiểm tra với pipeline
try:
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    test_inputs = [
        "chuan_hoa: mik k hieu j het",
        "chuan_hoa: t ko bik j lun",
        "chuan_hoa: cx muon di chs nhug k co time"
    ]
    print("\n📌 Kiểm tra với pipeline:")
    for test_input in test_inputs:
        result = pipe(
            test_input,
            max_new_tokens=128,
            num_beams=5,
            early_stopping=True,
            do_sample=False,
            no_repeat_ngram_size=2
        )
        print(f"🧪 Raw pipeline output: {result}")
        print(f"🛠 Input: {test_input}")
        print(f"✅ Output: {result[0]['generated_text'] if result else 'No output generated'}")
        print()
except Exception as e:
    print(f"❌ Lỗi khi chạy pipeline: {e}")

# Kiểm tra sinh văn bản thủ công
print("\n📌 Kiểm tra sinh văn bản thủ công:")
for test_input in test_inputs:
    try:
        input_ids = tokenizer(test_input, return_tensors="pt").input_ids
        output_ids = model.generate(
            input_ids,
            max_length=128,
            num_beams=10,
            early_stopping=True,
            do_sample=False,
            no_repeat_ngram_size=3
        )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"🛠 Input: {test_input}")
        print(f"✅ Manual output: {output}")
        print()
    except Exception as e:
        print(f"❌ Lỗi khi sinh văn bản thủ công: {e}")

# --------------------
# 9. Log quá trình huấn luyện
# --------------------
print("📉 Train Loss:", trainer.state.log_history)