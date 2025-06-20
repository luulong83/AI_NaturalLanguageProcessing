import numpy as np
from ripser import ripser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from stats_count import count_top_stats  # Giả định từ file gốc

# Khởi tạo PhoBERT
model_path = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)
model = model.to("cuda:1")
MAX_LEN = 128

# Hàm trích xuất attention weights (tương tự grab_attention_weights)
def grab_attention_weights_phobert(model, tokenizer, sentences, max_len, device):
    inputs = tokenizer.batch_encode_plus(
        sentences, return_tensors="pt", max_length=max_len, pad_to_max_length=True, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    attention = outputs.attentions  # (layers, batch, heads, seq_len, seq_len)
    return np.stack([layer.cpu().detach().numpy() for layer in attention])  # (layers, batch, heads, seq_len, seq_len)

# Hàm tính đặc trưng topological
def compute_topo_features(sentences, thresholds=[0.025, 0.05, 0.1, 0.25, 0.5, 0.75], stats_name="b0b1_e"):
    adj_matrices = grab_attention_weights_phobert(model, tokenizer, sentences, MAX_LEN, "cuda:1")
    ntokens = [min(len(tokenizer.tokenize(s)), MAX_LEN) for s in sentences]
    stats = count_top_stats(adj_matrices, thresholds, ntokens, stats_name.split("_"), stats_cap=500)
    return stats  # (layers, heads, thresholds, samples, features)

# Áp dụng
data = pd.read_csv("small_gpt_web/augmented_test_15k.csv")
sentences = data["sentence"].values[:100]  # Lấy 100 mẫu để demo
topo_features = compute_topo_features(sentences)  # (12, 12, 6, 100, 3) for b0b1_e
np.save("small_gpt_web/topo_features.npy", topo_features)