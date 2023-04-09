import numpy as np
import gzip
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import xlwt


# 载入数据
def load_data(path):
    with gzip.open(Path(path).as_posix(), 'rb') as f:
        ((train_X, train_y), (test_X, test_y), _) = pickle.load(f, encoding='latin-1')
    return train_X, train_y, test_X, test_y


# 随机选择进行批量梯度下降的索引
def get_batch(n, batch_size):
    batch_step = np.arange(0, n, batch_size)
    indices = np.arange(n, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i: i + batch_size] for i in batch_step]
    return batches


# 准确率计算
def accuracy(y_true, y_pred):
    return len(np.where(y_true==y_pred)[0]) / len(y_true)


# 绘制手写数字的灰度图
def show_image(data, index):
    plt.imshow(data.reshape((28, 28)), cmap='gray')
    plt.savefig(f'./Parameters_images/{index}.jpg')


# 绘制 Loss 曲线
def plot_loss(path, loss_train, loss_test):
    plt.figure(dpi=150)
    plt.title('Loss Curve')
    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.legend(['train', 'test'])
    plt.savefig(path+'LossCurve.jpg')


# 绘制 Accuracy 曲线
def plot_acc(path, acc):
    plt.figure(dpi=150)
    plt.title('Accuracy Curve')
    plt.plot(acc)
    plt.savefig(path+'AccCurve.jpg')


# 保存 loss and accuracy 的值
def save_metrics(path, train_loss, test_loss, acc):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    names = ['train_loss', 'test_loss', 'acc']
    for j in range(len(names)):
        sheet1.write(0, j, names[j])
        for i in range(len(acc)):
            sheet1.write(i+1, j, eval(names[j])[i])
    f.save(path+'metircs.xlsx')
