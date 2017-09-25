import csv
import mxnet as mx
import numpy as np

import logging
import argparse

PATH = "C:/Users/CamiXXX/Desktop/code/kaggel/Digital/"


def nn_train():
    """ MLP network architecture for MNIST
        全链接神经网络识别手写体 """

    # 训练数据读取：Read csv
    csv_reader = csv.reader(open(PATH+"train.csv", encoding='utf-8'))
    train_csv = []
    train_label = []
    # train的第一列是label, 后面42000列是数据
    for row in csv_reader:
        train_csv.append(row[1:])
        train_label.append(row[0])
    # Read csv using mxnet iter
    # train_csv和label的第一项都是csv的表头，不是数据，所以略过第一行
    train_iter = mx.io.NDArrayIter(np.asarray(train_csv[1:]), np.asarray(train_label[1:]), batch_size=20)
    # Debug 查看Iter
    # for batch in train_iter:
    #     print("batch_debug:")
    #     print([batch.data, batch.pad])

    # 测试数据读取
    csv_reader_test = csv.reader(open(PATH + "test.csv", encoding='utf-8'))
    test_csv = [row for row in csv_reader_test] # test没有label,直接取;但是第一行同样是表头，要略过
    test_iter = mx.io.NDArrayIter(np.asarray(test_csv[1:]), batch_size=20)


    # 网络搭建 Define the  MLP network.
    mlp = mx.symbol.Variable('data')
    mlp = mx.symbol.FullyConnected(mlp, name='fc1', num_hidden=128)
    mlp = mx.symbol.Activation(mlp, name='relu1', act_type="relu")
    mlp = mx.symbol.FullyConnected(mlp, name='fc2', num_hidden=64)
    mlp = mx.symbol.Activation(mlp, name='relu1', act_type="relu")
    mlp = mx.symbol.FullyConnected(mlp, name='fc3', num_hidden=10)
    mlp = mx.symbol.SoftmaxOutput(mlp, name='softmax')
    mx.viz.plot_network(mlp)
    print("Finish the network:"+str(mlp))

    # 建立Mod，并绑定网络mlp,并自己开始训练
    mod = mx.mod.Module(symbol=mlp,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])
    # # 申请空间：allocate memory given the input data and label shapes
    # mod.bind(data_shapes=train_iter.provide_data,
    #          label_shapes=train_iter.provide_label, )
    # # 初始化随机参数：initialize parameters by uniform random numbers
    # mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # # 使用学习率为0.1的随机梯度下降：use SGD with learning rate 0.1 to train
    # mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))
    # # 精确度：use accuracy as the metric
    # metric = mx.metric.create('acc')
    # # 训练：train 5 epochs, i.e. going over the data iter one pass
    # for epoch in range(5):
    #     train_iter.reset()
    #     metric.reset()
    #     for batch in train_iter:
    #         mod.forward(batch, is_train=True)  # compute predictions
    #         mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
    #         mod.backward()  # compute gradients
    #         mod.update()  # update parameters
    #     print('Epoch %d, Training %s' % (epoch, metric.get()))

    # 训练：使用mod.fit自动绑定和训练
    # construct a callback function to save checkpoints
    model_prefix = 'mx_mlp'
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint)

    # 预测
    predict_res = mod.predict(test_iter)
    # 处理预测结果
    test_label = []
    for each in predict_res:
        test_label.append(mx.ndarray.argmax(each,0))

    print(test_label)
    # print(predict_res[0:4])
    # score = mod.score(train_iter, ['acc'])
    # print("Accuracy score is %f" % (score[0][1]))

if __name__ == '__main__':
    nn_train()
