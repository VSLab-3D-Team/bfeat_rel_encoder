import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import time
import colorsys

"""
수업 진도를 빼보니, 너무 빠르다고 함.
Chapter당 한 주씩을 잡고 다음주까지를 Chapter 4를 잡는다. 
Chatper 5는 ~3/31
Chapter 6는 ~4/07

Solution은 모든 문제에 대해서 하고, 네이밍은 _(수정 날짜) 
Modified와 Additional에는 볼드체로 작성
솔루션까지 다 끝났을때, Chapter당 단톡방에 공지

- 뒷심(?) 
미국 유학, 석사는 국내에서 하는게 좋다.
석사는 돈이 많이 든다. 박사는 지원이 많이 된다. 
랩실 잘 알아보고, 2년동안 1년 반 때부터 석사
post master? ETRI에 면접 심사하는데 대부분 PoMa이다. 
다들 Top Conference는 다들 1~2개씩 가지고 있다.

CV에 깃헙 링크, 넣을 수 있는거는 다 넣어야한다.
- 포트폴리오, CV, 영어 성적, 성적표
- 깃헙에 정리

뉴립스, 매칭 시스템이 있다. 성능만 competitive하게 나와도 충분히 이 아이디어로 가능성이 있다.
"""

def make_color_map(num_classes=26):

    # HSV → RGB 변환 후, 0~1로 정규화된 RGB 배열 생성
    colors = []
    for i in range(num_classes):
        hue = i / num_classes  # hue를 균등하게 나눔 (0~1)
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.9)  # (hue, saturation, value)
        colors.append(rgb)

    colors_np = np.array(colors)  # (26, 3) numpy 배열
    return colors_np

def visualize_tsne_multilabel_rgb_wo_center(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    perplexity=50, n_iter=2000
):
    """
    Multi-Label 데이터를 RGB 색상으로 변환하여 t-SNE 시각화
    """
    # 🔹 PyTorch Tensor → NumPy 변환
    features = F.normalize(features, dim=1)
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 🔹 t-SNE 적용
    s = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features_np)
    e = time.time()
    print("Time for t-SNE dimension reductions: ", e - s)
    
    # 🔹 Multi-Label을 RGB 색상으로 변환 (각 클래스마다 고유 색상 지정)
    color_map = make_color_map()  # 각 클래스마다 무작위 색상 (RGB)
    colors = labels_np @ color_map  # Multi-Label을 RGB 색상으로 변환
    colors = np.clip(colors, 0, 1)  # RGB 범위 [0, 1]로 정규화
    
    label_counts = labels_np.sum(axis=1)  # 몇 개의 클래스를 갖는지 계산
    alpha_values = label_counts / label_counts.max()  # 정규화 (0~1 범위)
    alpha_values = np.clip(alpha_values, 0.05, 1)
    
    # 🔹 시각화
    fig, ax1 = plt.subplots(figsize=(13,10))
    ax1.scatter(features_2d[:, 0], features_2d[:, 1], color=colors, alpha=alpha_values)
    ax1.set_title("t-SNE Visualization of Multi-Label Features (RGB Blending)")
    ax1.set_xlabel("t-SNE Dim 1")
    ax1.set_ylabel("t-SNE Dim 2")
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig

def visualize_tsne_multilabel_rgb(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    centers: List[torch.Tensor],
    perplexity=50, n_iter=2000
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
    
    color_map = make_color_map()  # 각 클래스마다 무작위 색상 (RGB)
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

