import pandas as pd
import mxnet as mx
import numpy as np


def get_data():
    # 训练数据读取
    train_csv = pd.read_csv('../Titanic/train.csv', encoding='utf-8')
    test_csv = pd.read_csv('../Titanic/test.csv', encoding='utf-8')
    # 数据处理
    train_data = train_csv.drop(['PassengerId', 'Survived'], 1)
    train_label = train_csv.get('Survived')
    test_data = test_csv.drop('PassengerId', 1)
    train_data.fillna(value=0)
    test_data.fillna(value=0)
    return train_data, train_label, test_data


def csv_iter():
    train_iter = mx.io.CSVIter(data_csv='../Titanic/train.csv',
                               label_csv='../Titanic/train_clean.csv',
                               batch_size=20,
                               data_shape=(1,))
    test_iter = mx.io.CSVIter(data_csv='../Titanic/train.csv',
                               batch_size=20,
                               data_shape=(1,))
    return train_iter, test_iter


def mlp():
    """ 全链接网络搭建 Define the  MLP network."""
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=80)
    relu1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(relu1, name='fc2', num_hidden=64)
    relu2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(relu2, name='fc3', num_hidden=2)
    mlp = mx.symbol.SoftmaxOutput(fc3, name='softmax')
    mx.viz.plot_network(mlp)
    print("Finish the network.")
    return mlp


def train():
    train_iter, test_iter = get_iter()

    # 建立Mod，并绑定网络mlp,并自己开始训练
    mod = mx.mod.Module(symbol=mlp(),
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])
    # 申请空间：allocate memory given the input data and label shapes
    mod.bind(data_shapes=train_iter.provide_data,
             label_shapes=train_iter.provide_label, )
    # 初始化随机参数：initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # 使用学习率为0.1的随机梯度下降：use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))
    # 精确度：use accuracy as the metric
    metric = mx.metric.create('acc')
    # 训练：train 5 epochs, i.e. going over the data iter one pass
    for epoch in range(5):
        train_iter.reset()
        metric.reset()
        for batch in train_iter:
            mod.forward(batch, is_train=True)  # compute predictions
            mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
            mod.backward()  # compute gradients
            mod.update()  # update parameters
        print('Epoch %d, Training %s' % (epoch, metric.get()))

if __name__ == '__main__':
    train()
