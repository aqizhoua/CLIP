import matplotlib.pyplot as plt
import numpy as np

def draw_matrix(confusion_matrix,labels,label_order_list):
    # 模拟混淆矩阵
    # confusion_matrix = np.array([[0.9, 0.2, 0.1],
    #                              [0.2, 0.8, 0.3],
    #                              [0.1, 0.3, 0.7]])

    # labels=["dog","cat","bird"]


    # 设置颜色条的范围和颜色
    cmap = plt.cm.Reds  # 颜色映射
    norm = plt.Normalize(vmin=0, vmax=1)  # 颜色条的范围

    # 可视化混淆矩阵
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=cmap, norm=norm)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom")

    # 添加标签和网格线
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(label_order_list)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(confusion_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(confusion_matrix.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 添加文本标注
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black")

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.savefig("Confusion Matrix.png")
    plt.show()