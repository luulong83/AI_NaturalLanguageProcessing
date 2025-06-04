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
def create_text_point_cloud(sentences, model, max_words=100):
    word_vectors = []
    words = []  # Lưu từ để phân loại
    for sentence in sentences:
        tokens = preprocess_text(sentence)
        for word in tokens:
            if word in model and len(word_vectors) < max_words:
                word_vectors.append(model[word])
                words.append(word)
    return np.array(word_vectors), words

# Vẽ point cloud với phân loại (giảm chiều về 2D)
def plot_point_cloud(X, labels, title="Text Point Cloud with Clustering"):
    # Giảm chiều về 2D bằng PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    # Vẽ với màu khác nhau cho từng cụm
    plt.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1], s=10, color='blue', label='Cluster 1')
    plt.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1], s=10, color='red', label='Cluster 2')
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
    # Tải mô hình word2vec pre-trained
    print("Đang tải mô hình word2vec...")
    word2vec_model = load('glove-wiki-gigaword-100')  # Mô hình embedding 100 chiều

    # Dữ liệu đầu vào: các câu về thể thao và công nghệ
    sentences = [
        # Chủ đề thể thao
        "Football is a popular sport played worldwide.",
        "Basketball players dribble and shoot the ball.",
        "Tennis requires agility and strong serves.",
        # Chủ đề công nghệ
        "Artificial intelligence is transforming industries.",
        "Smartphones have advanced cameras and processors.",
        "Cloud computing enables scalable data storage."
    ]

    # Bài tập 1: Tạo point cloud từ văn bản
    text_cloud, words = create_text_point_cloud(sentences, word2vec_model, max_words=100)

    # Thêm phân loại bằng K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(text_cloud)

    # In các từ thuộc từng cụm
    print("Phân loại các từ:")
    for cluster in [0, 1]:
        print(f"\nCụm {cluster + 1}:")
        cluster_words = [words[i] for i in range(len(words)) if labels[i] == cluster]
        print(cluster_words)

    # Vẽ point cloud với phân loại
    text_cloud_2d = plot_point_cloud(text_cloud, labels, title="Text Point Cloud (Sports & Tech with Clustering)")

    # Bài tập 2: Persistent homology cho văn bản
    text_diagrams = compute_persistence_ripser(text_cloud, title="Persistence Diagram (Text)")

    # Nhận xét (in ra console)
    print("\nText Diagrams (Ripser):")
    print("H0:", text_diagrams[0])  # Các nhóm từ liên quan
    print("H1:", text_diagrams[1])  # Các vòng lặp trong không gian từ