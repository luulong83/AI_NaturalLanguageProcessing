# TDA Implementation Template
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.datasets import make_circles

class TDAAnalyzer:
    def __init__(self):
        self.data = None
        self.dgms = None
    
    def load_point_cloud(self, n_samples=100):
        """Tạo point cloud mẫu"""
        self.data, _ = make_circles(n_samples=n_samples, noise=0.1, factor=0.3)
        return self.data
    
    def compute_persistence(self, maxdim=1):
        """Tính persistent homology"""
        self.dgms = ripser(self.data, maxdim=maxdim)['dgms']
        return self.dgms
    
    def plot_results(self):
        """Vẽ kết quả"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Point cloud
        ax1.scatter(self.data[:, 0], self.data[:, 1])
        ax1.set_title('Point Cloud Data')
        
        # Persistence diagram
        plot_diagrams(self.dgms, ax=ax2)
        ax2.set_title('Persistence Diagram')
        
        plt.tight_layout()
        return fig
    
    def extract_features(self):
        """Trích xuất đặc trưng topo"""
        features = {}
        
        # Betti numbers
        features['betti_0'] = len(self.dgms[0])
        features['betti_1'] = len(self.dgms[1]) if len(self.dgms) > 1 else 0
        
        # Persistence statistics
        if len(self.dgms[0]) > 0:
            lifetimes_0 = self.dgms[0][:, 1] - self.dgms[0][:, 0]
            features['mean_lifetime_0'] = np.mean(lifetimes_0[lifetimes_0 != np.inf])
        
        if len(self.dgms) > 1 and len(self.dgms[1]) > 0:
            lifetimes_1 = self.dgms[1][:, 1] - self.dgms[1][:, 0]
            features['mean_lifetime_1'] = np.mean(lifetimes_1)
        
        return features

# Sử dụng
if __name__ == "__main__":
    tda = TDAAnalyzer()
    data = tda.load_point_cloud()
    dgms = tda.compute_persistence()
    features = tda.extract_features()
    
    print("Topological Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
    
    # Vẽ kết quả
    fig = tda.plot_results()
    plt.show()