# infer.py  (code dự đoán chuẩn hóa với model fine-tuned)

from transformers import T5Tokenizer, T5ForConditionalGeneration

def chuan_hoa_van_ban(text, model_path="./vit5_chuanhoa_model"):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    # model = T5ForConditionalGeneration.from_pretrained(model_path)
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Lỗi load mô hình: {e}")

    input_ids = tokenizer("chuan_hoa: " + text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128)
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
