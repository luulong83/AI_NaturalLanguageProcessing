import numpy as np
from sentence_transformers import SentenceTransformer
import gudhi as gd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

# Dữ liệu mẫu
texts = [
    "Sản phẩm rất tốt, tôi rất thích!",
    "Dịch vụ tệ, không đáng tiền.",
    "Món ăn ngon nhưng phục vụ chậm.",
    "Chất lượng ổn, giá hợp lý.",
    "Giao hàng nhanh, rất hài lòng!",
    "Không gian quán đẹp nhưng giá cao.",
    "Nhân viên thân thiện, dịch vụ tốt.",
    "Sản phẩm bình thường, không đặc biệt."
]
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
embeddings = model.encode(texts)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Kiểm tra khoảng cách
distances = pdist(embeddings_scaled)
print(f"Khoảng cách: min={min(distances):.3f}, max={max(distances):.3f}")

# Tạo Vietoris-Rips complex
max_edge_length = max(distances) * 1.5
rips_complex = gd.RipsComplex(points=embeddings_scaled, max_edge_length=max_edge_length)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
diag = simplex_tree.persistence()

# In persistence diagram
print("=== Persistence Diagram ===")
for dim, (birth, death) in diag:
    death_str = f"{death:.3f}" if np.isfinite(death) else "∞"
    print(f"H{dim}: Sinh tại {birth:.3f}, Chết tại {death_str}, Độ dai dẳng: {death - birth:.3f}")

# Chọn bình luận dựa trên H0 và H1
threshold = 0.01
selected_indices = []
for dim, (birth, death) in diag:
    if np.isfinite(death) and (death - birth) > threshold:
        # Tìm các điểm dữ liệu liên quan đến đặc trưng dai dẳng
        simplices = simplex_tree.get_simplices()
        for simplex, _ in simplices:
            if simplex_tree.filtration(simplex) <= death:
                selected_indices.extend(simplex)

# Loại bỏ trùng lặp và chọn bình luận
selected_indices = list(set(selected_indices))
selected_texts = [texts[i] for i in selected_indices if i < len(texts)]
print("\nBình luận được chọn để tăng cường:")
if selected_texts:
    for text in selected_texts:
        print(f"- {text}")
else:
    print("Không có bình luận nào được chọn.")