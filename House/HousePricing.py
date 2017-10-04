import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import mxnet as mx


def data_clean(item):
    """ Clean and modified """
    d = {
        'NA': 0.0,
        # MSZoning
        'RL': 0.0,
        'RM': 1.0,
        'C(all)': 2.0,
        'c(all)': 2.0,
        'FV': 3,
        'RH': 4,
        # Street
        'Pave': 1.0,
        # Alley
        'Grvl': 1.0,
        # LotShape
        'Reg': 0.0,
        'IR1': 1.0,
        'IR2': 2.0,
        'IR3': 3.0,
        # LandContour
        'Bnk': 1.0,
        'HLS': 2.0,
        'Low': 3.0,
        'Lvl': 4.0,
        # Utilities
        'AllPub': 1.0,
        'NoSeWa': 2.0,
        # LotConfig
        'Corner': 1.0,
        'CulDSac': 2.0,
        'FR2': 3.0,
        'FR3': 4.0,
        'Inside': 5.0,
        # LandSlope
        'Gtl': 1.0,
        'Mod': 2.0,
        'Sev': 3.0,

    }
    if item.isdigit:
        return float(item)
    elif d[item]:
        return d[item]
    else:
        return 0


def read_data():
    train_reader = pd.read_csv("../House/train.csv", encoding='utf-8')
    test_reader = pd.read_csv("../House/test.csv", encoding='utf-8')
    train_data = train_reader.drop("SalePrice", 1).values
    train_label = train_reader.get("SalePrice").values.tolist()
    train_iter = mx.io.NDArrayIter(np.array(train_data), np.array(train_label), batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(test_reader.values, batch_size=batch_size)

    return train_iter, test_iter


def nn_train():
    batch_size = 20

    train_iter = mx.io.CSVIter(data_csv="../House/train_data.csv",
                               label_csv="../House/train_label.csv",
                               batch_size=batch_size,
                               data_shape=(1, 79))

    test_iter = mx.io.CSVIter(data_csv="../House/test_data.csv",
                               batch_size=batch_size,
                               data_shape=(1, 79))


    # 网络搭建
    mlp = mx.symbol.Variable('data')
    mlp = mx.symbol.FullyConnected(mlp, name='fc1', num_hidden=80)
    mlp = mx.symbol.Activation(mlp, name='relu1', act_type="relu")
    mlp = mx.symbol.FullyConnected(mlp, name='fc2', num_hidden=40)
    mlp = mx.symbol.Activation(mlp, name='relu1', act_type="relu")
    mlp = mx.symbol.FullyConnected(mlp, name='fc3', num_hidden=1)
    mlp = mx.symbol.SoftmaxOutput(mlp, name='softmax')
    mx.viz.plot_network(mlp)

    # 建立Mod，并绑定网络mlp,
    mod = mx.mod.Module(symbol=mlp,
                        context=mx.gpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

    # 自己开始训练的写法
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
    # 全连接训练
    nn_train()
