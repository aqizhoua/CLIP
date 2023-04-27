import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_heatmap(confusion_matrix, labels,order_list):
    # 将混淆矩阵转换为矩阵数据框
    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=order_list)

    # 绘制热力图
    sns.set(font_scale=1.4)  # 设置字体大小
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="Blues")  # 绘制热力图

    plt.title("Confusion Matrix")  # 设置标题
    plt.xlabel("Predicted Labels")  # 设置x轴标签
    plt.ylabel("True Labels")  # 设置y轴标签
    plt.show()  # 显示热力图

# 测试代码
if __name__ == '__main__':
    confusion_matrix = np.array([[50, 10, 5], [10, 40, 10], [5, 10, 50]])
    labels = ["Class 1", "Class 2", "Class 3"]
    draw_heatmap(confusion_matrix, labels,[1,2,3])