import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# Bài tập 1: Tạo và trực quan hóa point cloud (vòng tròn với nhiễu)
def create_circle_point_cloud(n_points=100, noise=0.1):
    theta = np.linspace(0, 2*np.pi, n_points)
    
    return np.array(
        [np.cos(theta), np.sin(theta)]
    ).T + noise * np.random.randn(n_points, 2)


# Vẽ point cloud
def plot_point_cloud(X, title="Point Cloud"):
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()

# Bài tập 3: Tạo point cloud hình số 8
def create_figure_eight_point_cloud(n_points=200, noise=0.1):
    theta = np.linspace(0, 2*np.pi, n_points//2)
    # Vòng tròn trái
    X1 = np.array([np.cos(theta) - 1, np.sin(theta)]).T
    # Vòng tròn phải
    X2 = np.array([np.cos(theta) + 1, np.sin(theta)]).T
    return np.vstack([X1, X2]) + noise * np.random.randn(n_points, 2)

# Bài tập 2 & 4: Tính persistent homology với Ripser
def compute_persistence_ripser(X, title="Persistence Diagram"):
    diagrams = ripser(X, maxdim=1)['dgms']  # maxdim=1 để tính H0 và H1
    plot_diagrams(diagrams, show=True, title=title)
    return diagrams

# Chạy các bài tập
if __name__ == "__main__":
    # Bài tập 1: Vòng tròn
    circle_cloud = create_circle_point_cloud(n_points=100, noise=0.1)
    plot_point_cloud(circle_cloud, title="Circle Point Cloud")
    
    # Bài tập 2: Persistent homology cho vòng tròn (Ripser)
    circle_diagrams = compute_persistence_ripser(circle_cloud, title="Persistence Diagram (Circle)")
    
    # Bài tập 3: Hình số 8
    figure_eight_cloud = create_figure_eight_point_cloud(n_points=200, noise=0.1)
    plot_point_cloud(figure_eight_cloud, title="Figure Eight Point Cloud")
    figure_eight_diagrams = compute_persistence_ripser(figure_eight_cloud, title="Persistence Diagram (Figure Eight)")
    
    # Bài tập 4: Persistent homology cho vòng tròn (Ripser thay vì GUDHI)
    circle_ripser_diagram = compute_persistence_ripser(circle_cloud, title="Persistence Diagram (Circle, Ripser)")
    
    # Nhận xét (in ra console)
    print("Circle Diagrams (Ripser):")
    print("H0:", circle_diagrams[0])
    print("H1:", circle_diagrams[1])
    print("\nFigure Eight Diagrams (Ripser):")
    print("H0:", figure_eight_diagrams[0])
    print("H1:", figure_eight_diagrams[1])