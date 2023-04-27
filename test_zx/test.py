import torch
import clip
import numpy as np

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# 定义标签和数据集
labels = ['cat', 'dog', 'bird', 'fish']
dataset1 = [('cat_image1.jpg', 'cat'), ('dog_image1.jpg', 'dog'), ('bird_image1.jpg', 'bird')]
dataset2 = [('cat_image2.jpg', 'cat'), ('dog_image2.jpg', 'dog'), ('fish_image1.jpg', 'fish')]

# 提取标签的CLIP向量
label_features = []
for label in labels:
    text = clip.tokenize(label).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    label_features.append(text_features.cpu().numpy())

# 计算标签之间的相似度
similarity_matrix = np.zeros((len(labels), len(labels)))
for i in range(len(labels)):
    for j in range(len(labels)):
        if i == j:
            similarity_matrix[i][j] = 1.0
        else:
            similarity_matrix[i][j] = np.dot(label_features[i], label_features[j].T) / (np.linalg.norm(label_features[i]) * np.linalg.norm(label_features[j]))

# 打印标签之间的相似度矩阵
print(similarity_matrix)

# 将具有相似标签的数据样本聚合在一起
datasets = [dataset1, dataset2]
clustered_data = {}
for dataset in datasets:
    for image, label in dataset:
        for i, l in enumerate(labels):
            if label == l:
                if i not in clustered_data:
                    clustered_data[i] = []
                clustered_data[i].append((image, label))

# 打印聚合后的数据集
for i in range(len(labels)):
    print(f"Cluster {labels[i]}: {clustered_data[i]}")