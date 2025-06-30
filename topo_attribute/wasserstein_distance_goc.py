import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
from persim import wasserstein
from sklearn.preprocessing import StandardScaler
from underthesea import sentiment

# Dữ liệu mẫu
texts = [
    "Sản phẩm tuyệt vời!", "Dịch vụ tệ!", "Món ăn ngon, phục vụ chậm.",
    "Chất lượng tốt, giá hợp lý.", "Phục vụ không chuyên nghiệp!",
    "Thức ăn tuyệt, nhưng chờ lâu.", "Hàng giao nhanh, rất hài lòng.",
    "Giao hàng rất nhanh!", "Dịch vụ kém, không đáng tiền.",
    "Sản phẩm đẹp, dùng thích.", "Phục vụ chậm, thái độ tệ."
]

# Tính embeddings
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
embeddings = model.encode(texts)

# Tính điểm số cảm xúc
sentiments = [1 if sentiment(text) == 'positive' else -1 if sentiment(text) == 'negative' else 0 for text in texts]
sentiments = np.array(sentiments).reshape(-1, 1)
print("Sentiment scores:", sentiments)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Tính persistence diagram tham chiếu (toàn bộ dữ liệu)
reference_diagram = ripser(embeddings_scaled, maxdim=0)['dgms'][0]
reference_diagram = reference_diagram[reference_diagram[:, 1] != np.inf] if reference_diagram.size > 0 else np.array([])

# Tính Wasserstein distance cho từng bình luận
distances = []
selected_indices = []
threshold = 50.0  # Ngưỡng để chọn bình luận khác biệt
for i in range(len(texts)):
    diagram = ripser(embeddings_scaled[i:i+1], maxdim=0)['dgms'][0]
    diagram = diagram[diagram[:, 1] != np.inf] if diagram.size > 0 else np.array([])
    if len(diagram) > 0 and len(reference_diagram) > 0:
        dist = wasserstein(diagram, reference_diagram)
        distances.append((i, dist))
        if dist > threshold:
            selected_indices.append(i)

# In kết quả
print("\nBình luận được chọn để tăng cường:")
if selected_indices:
    for idx in selected_indices:
        print(f"- {texts[idx]} (Sentiment: {sentiments[idx][0]:.2f}, Wasserstein Distance: {distances[idx][1]:.2f})")
else:
    print("Không có bình luận nào được chọn (Wasserstein Distance nhỏ hơn ngưỡng).")

# Debug info
print("\nDebug info:")
print(f"Persistence Diagram (reference): {reference_diagram}")
for i, dist in distances:
    diagram = ripser(embeddings_scaled[i:i+1], maxdim=0)['dgms'][0]
    diagram = diagram[diagram[:, 1] != np.inf] if diagram.size > 0 else np.array([])
    print(f"Persistence Diagram (bình luận {i}): {diagram}")
print(f"Wasserstein distances: {distances}")