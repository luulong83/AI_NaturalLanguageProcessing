import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
from persim import wasserstein
from sklearn.preprocessing import StandardScaler
from underthesea import sentiment

# Dữ liệu mẫu (tăng số lượng bình luận)
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

# Tính điểm số cảm xúc bằng underthesea
sentiments = [1 if sentiment(text) == 'positive' else -1 if sentiment(text) == 'negative' else 0 for text in texts]
sentiments = np.array(sentiments).reshape(-1, 1)
print("Sentiment scores:", sentiments)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Phân loại tích cực/tiêu cực
positive_indices = [i for i, s in enumerate(sentiments) if s >= 0]
negative_indices = [i for i, s in enumerate(sentiments) if s < 0]
print("Positive indices:", positive_indices)
print("Negative indices:", negative_indices)

# Tính Persistence Diagrams (H0) cho hai nhóm
data_pos = embeddings_scaled[positive_indices]
data_neg = embeddings_scaled[negative_indices]

diagrams_pos_h0 = np.array([])
diagrams_neg_h0 = np.array([])
if len(data_pos) > 0:
    diagrams_pos_h0 = ripser(data_pos, maxdim=0)['dgms'][0]
    diagrams_pos_h0 = diagrams_pos_h0[diagrams_pos_h0[:, 1] != np.inf] if diagrams_pos_h0.size > 0 else np.array([])
if len(data_neg) > 0:
    diagrams_neg_h0 = ripser(data_neg, maxdim=0)['dgms'][0]
    diagrams_neg_h0 = diagrams_neg_h0[diagrams_neg_h0[:, 1] != np.inf] if diagrams_neg_h0.size > 0 else np.array([])

# Tính Wasserstein distance giữa nhóm tích cực và tiêu cực
distances = []
if len(diagrams_pos_h0) > 0 and len(diagrams_neg_h0) > 0:
    dist = wasserstein(diagrams_pos_h0, diagrams_neg_h0)
    distances.append((0, 1, dist))  # Đại diện cho nhóm tích cực vs. tiêu cực
    print(f"Wasserstein distance (H0) giữa nhóm tích cực và tiêu cực: {dist}")

# Chọn bình luận có khoảng cách lớn
threshold = 0.1
selected_indices = []
if distances:
    selected_indices = [i for i, j, dist in distances if dist > threshold]
    # Gán các bình luận từ nhóm tích cực/tiêu cực nếu khoảng cách lớn
    if selected_indices:
        selected_indices = positive_indices + negative_indices  # Chọn tất cả bình luận từ hai nhóm
print("Bình luận được chọn để tăng cường:")
for idx in set(selected_indices):
    print(f"- {texts[idx]} (Sentiment: {sentiments[idx][0]:.2f})")

# Debug info
print("\nDebug info:")
print(f"Persistence Diagram (H0, positive group): {diagrams_pos_h0}")
print(f"Persistence Diagram (H0, negative group): {diagrams_neg_h0}")
print(f"Wasserstein distances: {distances}")