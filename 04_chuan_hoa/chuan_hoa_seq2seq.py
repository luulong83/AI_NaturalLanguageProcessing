# chuan_hoa_seq2seq.py

from transformers import T5Tokenizer, T5ForConditionalGeneration

def chuan_hoa_van_ban(input_text):
    tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")
    model = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base")

    input_ids = tokenizer("chuanhoa: " + input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=50)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    input_text = "mik k hieu j het"
    output = chuan_hoa_van_ban(input_text)

    print("📝 Input:", input_text)
    print("✅ Output:", output)
