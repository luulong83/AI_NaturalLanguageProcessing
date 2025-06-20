import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from gensim.downloader import load
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from nltk.tokenize import word_tokenize
import nltk

# Tải NLTK tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Xóa dấu câu
    tokens = word_tokenize(text)  # Tách từ
    return tokens

# Tạo point cloud từ văn bản và lưu danh sách từ
def create_text_point_cloud(comments, model, max_words=100):
    word_vectors = []
    words = []  # Lưu từ để phân loại
    for comment in comments:
        tokens = preprocess_text(comment)
        for word in tokens:
            if word in model and len(word_vectors) < max_words:
                word_vectors.append(model[word])
                words.append(word)
    return np.array(word_vectors), words

# Vẽ point cloud với phân loại (giảm chiều về 2D)
def plot_point_cloud(X, labels, title="Sentiment Point Cloud with Clustering"):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1], s=10, color='green', label='Cluster 1 (Positive)')
    plt.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1], s=10, color='red', label='Cluster 2 (Negative)')
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.axis("equal")
    plt.show()
    return X_2d

# Tính persistent homology với Ripser
def compute_persistence_ripser(X, title="Persistence Diagram"):
    diagrams = ripser(X, maxdim=1)['dgms']  # maxdim=1 để tính H0 và H1
    plot_diagrams(diagrams, show=True, title=title)
    return diagrams

# Chạy ví dụ
if __name__ == "__main__":
    # Tải mô hình GloVe pre-trained
    print("Đang tải mô hình GloVe...")
    glove_model = load('glove-wiki-gigaword-100')  # Mô hình embedding 100 chiều

    # Dữ liệu đầu vào: bình luận về sản phẩm
    comments = [
        # Bình luận tích cực
        "This phone is amazing with great battery life.",
        "Love the camera quality and fast performance.",
        "Really happy with this purchase, excellent product.",
        # Bình luận tiêu cực
        "The phone overheats and has poor battery.",
        "Disappointed with the slow processor and bad camera.",
        "Not worth the price, terrible customer service."
    ]

    # Tạo point cloud từ bình luận
    text_cloud, words = create_text_point_cloud(comments, glove_model, max_words=100)

    # Phân loại bằng K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(text_cloud)

    # In các từ thuộc từng cụm
    print("Phân loại các từ theo cảm xúc:")
    for cluster in [0, 1]:
        print(f"\nCụm {cluster + 1} {'(Tích cực)' if cluster == 0 else '(Tiêu cực)'}:")
        cluster_words = [words[i] for i in range(len(words)) if labels[i] == cluster]
        print(cluster_words)

    # Vẽ point cloud với phân loại
    text_cloud_2d = plot_point_cloud(text_cloud, labels, title="Sentiment Point Cloud (Positive vs Negative)")

    # Tính persistent homology
    text_diagrams = compute_persistence_ripser(text_cloud, title="Persistence Diagram (Sentiment)")

    # In kết quả topological
    print("\nSentiment Diagrams (Ripser):")
    print("H0:", text_diagrams[0])  # Các nhóm từ liên quan
    print("H1:", text_diagrams[1])  # Các vòng lặp trong không gian từ