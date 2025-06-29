import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

# Dữ liệu mẫu: bình luận tiếng Việt
texts = [
    "Sản phẩm rất tốt, tôi rất thích!",
    "Dịch vụ tệ, không đáng tiền.",
    "Món ăn ngon nhưng phục vụ chậm."
]

# Tạo nhúng văn bản
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
embeddings = model.encode(texts)

# Tính persistence diagram với Ripser
diagrams = ripser(embeddings, maxdim=1)['dgms']

# In thông tin H0 và H1
print("=== Persistence Diagrams ===")
for i, dgm in enumerate(diagrams):
    print(f"\nH{i} có {len(dgm)} điểm:")
    for idx, pt in enumerate(dgm):
        birth, death = pt
        death_str = f"{death:.3f}" if np.isfinite(death) else "∞"
        print(f"  Điểm {idx+1}: Sinh tại {birth:.3f}, Chết tại {death_str}")

# Trực quan hóa persistence diagram
fig, ax = plt.subplots()
plot_diagrams(diagrams, ax=ax, show=False)
plt.title("Persistence Diagram của Nhúng Văn Bản")

# Ghi chú điểm H0 trên biểu đồ (chỉ nếu muốn dễ hình dung)
H0 = diagrams[0]
for idx, (birth, death) in enumerate(H0):
    ax.annotate(f"H0-{idx+1}", xy=(birth, death), xytext=(birth+0.02, death+0.02),
                arrowprops=dict(arrowstyle="->", color='gray'), fontsize=8, color='blue')

# Nếu có điểm H1, chú thích thêm (hiếm thấy với ít văn bản)
H1 = diagrams[1]
for idx, (birth, death) in enumerate(H1):
    ax.annotate(f"H1-{idx+1}", xy=(birth, death), xytext=(birth+0.02, death+0.02),
                arrowprops=dict(arrowstyle="->", color='gray'), fontsize=8, color='orange')

plt.show()
