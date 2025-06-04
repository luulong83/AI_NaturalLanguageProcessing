import matplotlib.pyplot as plt # đễ vẽ đồ thị
from sklearn.feature_extraction.text import TfidfVectorizer #chuyễn văn bản thành vector
from sklearn.naive_bayes import MultinomialNB # Mô hình học máy Naive Bayes
from sklearn.pipeline import Pipeline # xử lý chuỗi các bước
from sklearn.model_selection import train_test_split # chia dữ liệu thành train và test
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay # đánh giá mô hình

# ------------------------------------------------------
# Bước 2: Dữ liệu mẫu
# ------------------------------------------------------
# Đây là danh sách các câu bình luận tiếng việt
# Chúng ta gán nhãn: "pos" = tích cực, "neg" = tiêu cực
def main():
    documents =[
        "Sản phẩm tuyệt vời, chất lượng rất tốt",         # tích cực
        "Tôi rất hài lòng với dịch vụ",                   # tích cực
        "Giao hàng nhanh, đóng gói cẩn thận",             # tích cực
        "Đóng gói chắc chắn, giao hàng đúng hẹn",         # tích cực
        "Tôi thích sản phẩm này",                         # tích cực
        "Sản phẩm rất đẹp, đúng mô tả",                   # tích cực

        "Quá tệ, sản phẩm lỗi hoàn toàn",                 # tiêu cực
        "Dịch vụ quá chậm, không chuyên nghiệp",          # tiêu cực
        "Tôi thất vọng về sản phẩm này",                  # tiêu cực
        "Giao hàng sai, sản phẩm không giống hình",       # tiêu cực
        "Chất lượng kém, không đáng tiền",                # tiêu cực
        "Dịch vụ khách hàng không hỗ trợ gì cả",          # tiêu cực
    ]

    # Danh sách nhãn tương ứng với từng câu
    labels = ['pos'] * 6 +  ['neg'] *6 # tạo 6 'pos' và 6 'neg'

    # -------------------------------------
    # Bước 3: Chia dữ liệu train / test
    # -------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.3,random_state=42)
    # - 70% dùng để huấn luyện (train), 30% để kiễm tra (test)

    # -------------------------------------
    # Bước 4: Xây dụng Pipeline
    # -------------------------------------
    # Pipeline này bao gồm 2 bước:
    # 1. TF-IDF Vectorizer: chuyển văn bản thành vector số
    # 2. MultiomialNB: mô hình Naive Bayes để học
    pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()), # bước 1: chuyển từ sang số
            ( 'clf', MultinomialNB())     # bước 2: mô hình phân loại
    ])

    # 🚀 Huấn luyện mô hình
    pipeline.fit(X_train, y_train)
    
    # 🧪 Đánh giá trên tập test
    y_pred = pipeline.predict(X_test)
    print("Kết quả dự đoán trên tập test:")
    print(classification_report(y_test, y_pred, target_names=['neg', 'pos']))
    
    # Hiển thị ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred, labels=['neg', 'pos'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['neg', 'pos'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Ma trận nhầm lẫn")
    plt.show()
    
     # 🔍 Dự đoán cảm xúc với dữ liệu mới
    new_comments = [
        "Tôi rất thích sản phẩm này",
        "Dịch vụ quá tệ, không đáng tiền",
        "Sản phẩm ổn trong tầm giá",
        "Mình sẽ không mua lần sau"
    ]
    predictions = pipeline.predict(new_comments)

    print("\n=== 🤖 Dự đoán cảm xúc bình luận mới ===")
    for comment, label in zip(new_comments, predictions):
        print(f"📝 Bình luận: \"{comment}\" ➜ 🧭 Dự đoán: {label}")
        
# 🚪 Chạy chương trình  
if __name__ == "__main__":
    print("chào các bạn, chương trình đang chạy...")
    # -------------------------------------
    main()