import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Dữ liệu mẫu
texts = [
    "Sản phẩm rất tốt, tôi rất thích!",
    "Dịch vụ tệ, không đáng tiền.",
    "Món ăn ngon nhưng phục vụ chậm."
]

# Nhúng câu
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
embeddings = model.encode(texts)

# ----- 1. In khoảng cách cosine -----
print("\n=== Cosine Distances ===")
n = len(texts)
for i in range(n):
    for j in range(i+1, n):
        dist = cosine_distances([embeddings[i]], [embeddings[j]])[0][0]
        print(f"Khoảng cách giữa câu {i+1} và câu {j+1}: {dist:.4f}")

# ----- 2. PCA trực quan hóa -----
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, point in enumerate(reduced):
    plt.scatter(point[0], point[1], label=f"Câu {i+1}")
    plt.text(point[0] + 0.01, point[1] + 0.01, f"{i+1}: {texts[i]}", fontsize=9)

plt.title("Biểu diễn PCA của các câu")
plt.xlabel("Thành phần chính 1")
plt.ylabel("Thành phần chính 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
