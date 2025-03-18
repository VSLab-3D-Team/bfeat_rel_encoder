import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import time

def visualize_tsne_multilabel_rgb(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    centers: List[torch.Tensor],
    perplexity=100, n_iter=2500
):
    """
    Multi-Label 데이터를 RGB 색상으로 변환하여 t-SNE 시각화
    """
    # 🔹 PyTorch Tensor → NumPy 변환
    features = F.normalize(features, dim=1)
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    extra_points_np = torch.cat(centers, dim=0).cpu().numpy()

    combined_data = np.vstack([features_np, extra_points_np])
    # 🔹 t-SNE 적용
    s = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    combined_tsne = tsne.fit_transform(combined_data)
    e = time.time()
    print("Time for t-SNE dimension reductions: ", e - s)
    
    features_2d = combined_tsne[:features_np.shape[0]]  # (B, 2)
    center_2d = combined_tsne[features_np.shape[0]:] # (C + 1, 2)

    # 🔹 Multi-Label을 RGB 색상으로 변환 (각 클래스마다 고유 색상 지정)
    num_classes = labels_np.shape[1]
    centers_label = torch.eye(num_classes).cpu().numpy()
    
    color_map = np.random.rand(num_classes, 3)  # 각 클래스마다 무작위 색상 (RGB)
    colors = labels_np @ color_map  # Multi-Label을 RGB 색상으로 변환
    colors = np.clip(colors, 0, 1)  # RGB 범위 [0, 1]로 정규화
    
    label_counts = labels_np.sum(axis=1)  # 몇 개의 클래스를 갖는지 계산
    alpha_values = label_counts / label_counts.max()  # 정규화 (0~1 범위)
    alpha_values = np.clip(alpha_values, 0.05, 1)
    
    
    c_colors = centers_label @ color_map
    c_colors = np.clip(c_colors, 0, 1)
    c_colors = np.concatenate([c_colors, np.array([[0., 0., 0.]])], axis=0)

    # 🔹 시각화
    fig, ax1 = plt.subplots(figsize=(13,10))
    ax1.scatter(features_2d[:, 0], features_2d[:, 1], color=colors, alpha=alpha_values)
    ax1.scatter(center_2d[:, 0], center_2d[:, 1], c=c_colors, marker='*', s=150, label="center Points")
    ax1.set_title("t-SNE Visualization of Multi-Label Features (RGB Blending)")
    ax1.set_xlabel("t-SNE Dim 1")
    ax1.set_ylabel("t-SNE Dim 2")
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    return fig

def visualize_tsne_3d(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    perplexity=30, n_iter=1000
):
    """
    512D Feature를 3D t-SNE로 변환하여 시각화
    """
    features_np = features.cpu().numpy()  # (B, 512) → NumPy 변환
    labels_np = labels.cpu().numpy()

    # 🔹 t-SNE 적용 (512D → 3D)
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_3d = tsne.fit_transform(features_np)  # (B, 3)

    # 🔹 Multi-Label을 RGB 색상으로 변환 (각 클래스마다 고유 색상 지정)
    num_classes = labels_np.shape[1]
    color_map = np.random.rand(num_classes, 3)  # 각 클래스마다 무작위 색상 (RGB)
    colors = labels_np @ color_map  # Multi-Label을 RGB 색상으로 변환
    colors = np.clip(colors, 0, 1)  # RGB 범위 [0, 1]로 정규화

    # 🔹 3D 시각화
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=colors, cmap='jet', alpha=0.8)
    fig.colorbar(scatter)

    ax.set_title("3D t-SNE Visualization of 512-D Features")
    plt.savefig("tsne_3d_visualization.png", dpi=300, bbox_inches='tight')

