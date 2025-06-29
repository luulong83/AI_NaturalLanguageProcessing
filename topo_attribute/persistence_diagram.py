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

# Trực quan hóa persistence diagram
plot_diagrams(diagrams, show=True)
plt.title("Persistence Diagram của Nhúng Văn Bản")
plt.show()