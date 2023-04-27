import torch
import clip
import numpy as np

from read_excel import read_excel
from draw import draw_matrix


list_data = read_excel()
label_order_list=[]
labels_list=[]
for i,line in enumerate(list_data):
    label_order_list.extend([i]*len(line))
    labels_list.extend(line)



# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# 提取标签的CLIP向量
label_features = []
for label in labels_list:
    label_sentence = "a centered satellite photo of {}".format(label)
    text = clip.tokenize(label_sentence).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    label_features.append(text_features.cpu().numpy())

# 计算标签之间的相似度
similarity_matrix = np.zeros((len(labels_list), len(labels_list)))
for i in range(len(labels_list)):
    for j in range(len(labels_list)):
        if i == j:
            similarity_matrix[i][j] = 1.0
        else:
            similarity_matrix[i][j] = np.dot(label_features[i], label_features[j].T) / (np.linalg.norm(label_features[i]) * np.linalg.norm(label_features[j]))

# 打印标签之间的相似度矩阵
print(similarity_matrix)
print(similarity_matrix.shape)
draw_matrix(similarity_matrix,labels_list,label_order_list)