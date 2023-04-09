import numpy as np
from train import NeuralNetwork
from utils import accuracy

# 数据载入
features = np.load('./test/features.npy') # 特征（行为每个样本，列为784个特征的矩阵）
labels = np.load('./test/labels.npy') # 样本的真实标签

# 模型载入
model = NeuralNetwork(784, 200, 10, 0.001)
model.load('./output/lr0.1_hd200_ri0.001/parameters.npz')

# 模型预测
model(features)

# 输出准确率
print('预测准确率：', accuracy(labels, model.y_pred), '.')
