from transformers import T5Tokenizer, T5ForConditionalGeneration

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
        num_beams=10,  # Tăng beams để cải thiện chất lượng
        early_stopping=True,
        do_sample=False,
        no_repeat_ngram_size=3  # Tránh lặp từ
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

if __name__ == "__main__":
    test_sentences = [
        "mik k hieu j het",
        "t ko bik j lun",
        "cx muon di chs nhug k co time"
    ]

    for sent in test_sentences:
        print(f"📝 Input: {sent}")
        print(f"✅ Output: {chuan_hoa_van_ban(sent)}\n")