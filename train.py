import numpy as np
import math
import os
from pathlib import Path
from utils import *


# 神经网络
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, regularization_intensity):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.regularization_intensity = regularization_intensity

        self.w1 = np.random.randn(self.hidden_dim, self.input_dim)
        self.b1 = np.random.randn(self.hidden_dim)
        self.w2 = np.random.randn(self.output_dim, self.hidden_dim)
        self.b2 = np.random.randn(self.output_dim)

    # 正向传播
    def __call__(self, data):
        self.x = data

        self.z1 = np.add(np.dot(self.w1, data.T), self.b1.reshape(-1, 1))
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.add(np.dot(self.w2, self.a1), self.b2.reshape(-1, 1))
        self.a2 = self.softmax(self.z2.T)
        self.y_pred = np.argmax(self.a2, axis=1)

        return self.y_pred

    # sigmoid 激活函数
    @staticmethod
    def sigmoid(x):
        if np.all(x >= 0):
            return 1. / (1.+np.exp(-x))
        else:
            return np.exp(x) / (1. + np.exp(x))

    # softmax 激活函数
    @staticmethod
    def softmax(x):
        return np.divide(np.exp(x), np.sum(np.exp(x), axis=1).reshape(-1, 1))

    # loss 计算
    def loss(self, y_true):
        self.y_true = y_true
        n = len(y_true)
        loss = -np.sum(np.log(self.a2[np.arange(n), y_true])) / n
        return loss

    # 反向传播
    def backward(self, lr_start, epoch):
        lr_end = 0.0001
        lr_decay = 200
        lr = lr_end + (lr_start - lr_end) * math.exp(-epoch/lr_decay)

        self.z2_grad = self.a2.copy()
        n = len(self.y_true)
        self.z2_grad[np.arange(n), self.y_true] -= 1

        self.w2_grad = self.z2_grad.T.reshape(self.output_dim, 1, len(self.y_true)) * self.a1
        self.b2_grad = self.z2_grad.T

        self.z1_grad = np.dot(self.z2_grad, self.w2).T * self.a1 * (1-self.a1)
        self.w1_grad = self.z1_grad.reshape(self.hidden_dim, 1, len(self.y_true)) * self.x.T
        self.b1_grad = self.z1_grad

        self.w2_grad = np.sum(self.w2_grad, axis=2) + 2 * self.regularization_intensity * self.w2
        self.w1_grad = np.sum(self.w1_grad, axis=2) + 2 * self.regularization_intensity * self.w1
        self.b2_grad = np.sum(self.b2_grad, axis=1) + 2 * self.regularization_intensity * self.b2
        self.b1_grad = np.sum(self.b1_grad, axis=1) + 2 * self.regularization_intensity * self.b1

        self.w2 -= lr * self.w2_grad
        self.b2 -= lr * self.b2_grad
        self.w1 -= lr * self.w1_grad
        self.b1 -= lr * self.b1_grad

    # 保存模型
    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    # 加载模型
    def load(self, path):
        parameters = np.load(path)
        self.w1 = parameters['w1']
        self.b1 = parameters['b1']
        self.w2 = parameters['w2']
        self.b2 = parameters['b2']



# 配置参数
class Config:
    def __init__(self):
        self.seed = 1 # numpy的随机数种子
        self.epoches = 200 # 训练轮次
        self.batch_size = 64 # 批量梯度下降中每批数量
        self.lr_start = 0.1 # 学习率的初始值
        self.hidden_dim = 256 # 隐藏层层数
        self.regularization_intensity = 0.005 # 正则化强度


# 训练函数
def train(cfg):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    model_path = f'{curr_path}/output/lr{cfg.lr_start}_hd{cfg.hidden_dim}_ri{cfg.regularization_intensity}/'

    np.random.seed(cfg.seed)
    train_X, train_y, test_X, test_y = load_data('./data/mnist.pkl.gz')
    n, input_dim = train_X.shape
    model = NeuralNetwork(input_dim, cfg.hidden_dim, 10, cfg.regularization_intensity)
    loss_train_total = []
    loss_test_total = []
    acc_total = []
    print(f'学习率起始值:{cfg.lr_start}, 隐藏层层数:{cfg.hidden_dim}, 正则化强度:{cfg.regularization_intensity}')
    for epoch in range(cfg.epoches):
        batch_indices = get_batch(len(test_X), cfg.batch_size)
        batch_num = 0
        loss_epoch = 0

        for batch in batch_indices:
            batch_num += 1
            train_X_batch = train_X[batch]
            model(train_X_batch)
            y_true = train_y[batch]
            loss_batch = float(model.loss(y_true))
            loss_epoch += 1 / len(batch_indices) * (loss_batch - loss_epoch)
            model.backward(cfg.lr_start, epoch)

        loss_train_total.append(loss_epoch)

        test_y_predict = model(test_X)
        test_loss = model.loss(test_y)
        loss_test_total.append(test_loss)

        acc = accuracy(test_y, test_y_predict)
        acc_total.append(acc)

        print(f'epoch:{epoch+1}/{cfg.epoches}\t   train_loss:{round(loss_epoch, 2)}\t  test_loss:{round(test_loss, 2)}\t acc:{np.round(acc*100, 2)}%.')

    Path(model_path).mkdir(parents=True, exist_ok=True) # 创建文件夹
    plot_loss(model_path, loss_train_total, loss_test_total) # 绘制loss曲线
    plot_acc(model_path, acc_total) # 绘制accuracy曲线
    save_metrics(model_path, loss_train_total, loss_test_total, acc_total) # 保存loss, accuracy 在每个epoch的值
    model.save(model_path+'parameters') # 保存模型参数

    print('='*25,'模型保存完成', '='*25)


if __name__ == '__main__':
    hyperparameters = ['lr', 'hidden_dim', 'regularization_intensity'] # 三个需要进行网格搜索的超参数

    grid = {
        'lr': [0.1, 0.01, 0.001], # 学习率
        'hidden_dim': [50, 100, 200], # 隐藏层
        'regularization_intensity': [0.1, 0.01, 0.001] # 正则化强度
    }

    # 进行 grid search
    for lr_start in grid['lr']:
        for hidden_dim in grid['hidden_dim']:
            for regularization_intensity in grid['regularization_intensity']:
                cfg = Config()
                cfg.lr_start = lr_start
                cfg.hidden_dim = hidden_dim
                cfg.regularization_intensity = regularization_intensity
                train(cfg)
