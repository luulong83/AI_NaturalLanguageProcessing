import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import random
import py_vncorenlp

# Khởi tạo VnCoreNLP cho từ điển đồng nghĩa
py_vncorenlp.download_model()
annotator = py_vncorenlp.VnCoreNLP()

# Hàm thay thế từ đồng nghĩa
def synonym_replacement(sentence):
    words = annotator.tokenize(sentence)[0]
    new_words = words.copy()
    for i, word in enumerate(words):
        synonyms = annotator.get_synonyms(word)  # Giả định VnCoreNLP có hàm này
        if synonyms and random.random() < 0.3:  # Thay thế ngẫu nhiên 30% từ
            new_words[i] = random.choice(synonyms)
    return " ".join(new_words)

# Hàm back-translation
def back_translation(sentence):
    model_name = "Helsinki-NLP/opus-mt-vi-en"
    tokenizer_en = MarianTokenizer.from_pretrained(model_name)
    model_en = MarianMTModel.from_pretrained(model_name)
    model_name = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer_vi = MarianTokenizer.from_pretrained(model_name)
    model_vi = MarianMTModel.from_pretrained(model_name)
    
    inputs = tokenizer_en(sentence, return_tensors="pt", truncation=True, max_length=128)
    translated = model_en.generate(**inputs)
    en_text = tokenizer_en.decode(translated[0], skip_special_tokens=True)
    
    inputs = tokenizer_vi(en_text, return_tensors="pt", truncation=True, max_length=128)
    translated = model_vi.generate(**inputs)
    return tokenizer_vi.decode(translated[0], skip_special_tokens=True)

# Hàm tăng cường dữ liệu
def augment_data(data):
    augmented_data = []
    for _, row in data.iterrows():
        sentence = row["sentence"]
        label = row["label"]
        augmented_data.append({"sentence": sentence, "label": label})
        
        # Thay thế từ đồng nghĩa
        augmented_data.append({"sentence": synonym_replacement(sentence), "label": label})
        # Back-translation
        augmented_data.append({"sentence": back_translation(sentence), "label": label})
    return pd.DataFrame(augmented_data)

# Áp dụng
data = pd.read_csv("small_gpt_web/test_5k.csv")
augmented_data = augment_data(data)
augmented_data.to_csv("small_gpt_web/augmented_test_15k.csv", index=False)