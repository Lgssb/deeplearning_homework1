# deeplearning_homework1 构建两层神经网络分类器

### 文件说明
data： mnist数据集

train.py：构建两层神经网络，网路训练，grid-search超参数优化

visualize.py：参数可视化

test.py： 模型测试，输出分类精度

utils.py：辅助函数，如文件读取，绘图等

### 训练
打开终端cd到项目根目录，并运行

```bash
python train.py
```

生成output文件夹，对于各种超参数搭配都生成独立的子文件夹，下有四个文件，

“LossCurve.jpg”为Loss曲线；

“AccCurve.jpg”为Accuracy曲线；

“metrics.xlsx”为每个训练epoch下的训练集损失、测试集损失和在测试集上的预测准确率；

“parameters.npz”为训练完成后保存的模型参数。

### 测试
需新建test文件夹，将需要测试的数据分为 “features.npy" 和 "labels.npy" 放入 test文件夹中，

- features.npy 存储特征，行为样本量，列为784个特征；
- labels.npy 存储真实标签。

打开终端cd到项目根目录，之后运行：

```bash
python test.py
```

即可输出预测的准确率。
