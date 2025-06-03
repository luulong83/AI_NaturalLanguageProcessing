from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Dùng mô hình NER tiếng Việt (VinAI đã huấn luyện sẵn)
model_name = "vinai/phobert-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained("NlpHUST/NER-PhoBERT-base")

# Tạo pipeline NER
ner = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

text = """
Hôm nay, Thủ tướng Phạm Minh Chính đã đến Hà Nội và gặp đại diện của Vingroup và Google.
"""

# Thực hiện trích xuất
entities = ner(text)

# In kết quả
for e in entities:
    print(f"{e['entity_group']}: {e['word']} ({e['score']:.2f})")
