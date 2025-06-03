from transformers import T5Tokenizer, T5ForConditionalGeneration
from underthesea import pos_tag, word_tokenize
import pandas as pd

# --------------------
# 1. Hàm chuẩn hóa văn bản (từ infer.py)
# --------------------
def chuan_hoa_van_ban(text, model_path="./vit5_chuanhoa_model"):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Lỗi load mô hình hoặc tokenizer: {e}")
        return None

    input_ids = tokenizer("chuan_hoa: " + text, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_length=128,
        num_beams=10,
        early_stopping=True,
        do_sample=False,
        no_repeat_ngram_size=3
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

# --------------------
# 2. Hàm gán nhãn từ loại
# --------------------
def gan_nhan_tu_loai(sentence):
    try:
        # Tách từ bằng Underthesea
        tokenized_sentence = word_tokenize(sentence, format="text")
        # Gán nhãn từ loại
        pos_tags = pos_tag(sentence)
        return pos_tags
    except Exception as e:
        print(f"❌ Lỗi khi gán nhãn từ loại: {e}")
        return None

# --------------------
# 3. Hàm kết hợp chuẩn hóa và gán nhãn từ loại
# --------------------
def process_sentence(text):
    print(f"📝 Input gốc: {text}")
    
    # Chuẩn hóa văn bản
    normalized_text = chuan_hoa_van_ban(text)
    print(f"✅ Văn bản chuẩn hóa: {normalized_text if normalized_text else text}")
    
    # Gán nhãn từ loại
    pos_tags = gan_nhan_tu_loai(normalized_text if normalized_text else text)
    print(f"✅ Nhãn từ loại: {pos_tags}")
    return normalized_text, pos_tags

# --------------------
# 4. Chạy thử với các câu
# --------------------
if __name__ == "__main__":
    test_sentences = [
        "Tôi đang học NLP bằng tiếng Việt.",  # Câu đã chuẩn
        "mik k hieu j het",                  # Câu cần chuẩn hóa
        "t ko bik j lun",
        "cx muon di chs nhug k co time"
    ]

    for sentence in test_sentences:
        print("\n" + "="*50)
        normalized, pos_tags = process_sentence(sentence)
        print("="*50 + "\n")