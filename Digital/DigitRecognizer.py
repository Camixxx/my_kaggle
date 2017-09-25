# coding=utf-8
import csv
import os
import mxnet as mx
import numpy as np

import logging
import argparse

# PATH = "C:/Users/CamiXXX/Desktop/code/kaggel"
PATH = os.getcwd()


def nn_train():
    """ MLP network architecture for MNIST
        全链接神经网络识别手写体 """

    # 训练数据读取：Read csv
    csv_reader = csv.reader(open(PATH + "/Digital/train.csv", encoding='utf-8'))
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
    csv_reader_test = csv.reader(open(PATH + "/Digital/test.csv", encoding='utf-8'))
    test_csv = [row for row in csv_reader_test]  # test没有label,直接取;但是第一行同样是表头，要略过
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
    print("Finish the network:" + str(mlp))

    # 建立Mod，并绑定网络mlp,
    mod = mx.mod.Module(symbol=mlp,
                        context=mx.gpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

    # # 自己开始训练的写法
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
    mod.fit(train_iter, num_epoch=15, epoch_end_callback=checkpoint)

    # 预测
    predict_res = mod.predict(test_iter, 20)
    # 处理预测结果
    test_label = []
    for each in predict_res:
        test_label.append(int(mx.ndarray.argmax(each, 0).asscalar()))

    print(test_label[0:1])
    # print(predict_res[0:4])
    # score = mod.score(train_iter, ['acc'])
    # print("Accuracy score is %f" % (score[0][1]))

    # Write
    with open("Digital/res_epoch15.csv", "w") as resFile:
        csv_writer = csv.writer(resFile)
        csv_writer.writerow(['ImageId', 'Label'])
        index = 1
        for label in test_label:
            csv_writer.writerow([index, label])
            index += 1

    return


def cnn_train():
    """ MLP network architecture for MNIST
        CNN神经网络识别手写体 """

    batch_size = 20

    # 训练数据读取：Read csv
    csv_reader = csv.reader(open("Digital/train.csv", encoding='utf-8'))
    train_csv = []
    train_label = []
    # train的第一列是label, 后面42000列是数据
    for row in csv_reader:
        train_csv.append(np.asarray(row[1:]).reshape(1, 28, 28))  # make it to 4d
        # train_csv.append(np.asarray(row[1:]))
        train_label.append(row[0])
    # Read csv using mxnet iter
    # train_csv和label的第一项都是csv的表头，不是数据，所以略过第一行
    train_iter = mx.io.NDArrayIter(np.asarray(train_csv[1:]),
                                   np.asarray(train_label[1:]), batch_size=batch_size)
    # Debug 查看Iter
    # for batch in train_iter:
    #     print("batch_debug:")
    #     print([batch.data, batch.pad])

    # 测试数据读取
    csv_reader_test = csv.reader(open("Digital/test.csv", encoding='utf-8'))
    test_csv = [np.asarray(row).reshape(1, 28, 28) for row in csv_reader_test]  # test没有label,直接取;但是第一行同样是表头，要略过
    test_iter = mx.io.NDArrayIter(np.asarray(test_csv[1:]), batch_size=batch_size)

    # 网络搭建 Define the  CNN network.
    data = mx.symbol.var('data')
    # first conv layer
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # second conv layer
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # first fullc layer
    # flatten = mx.symbol.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=pool2, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # softmax loss
    cnnnet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    print("Finish the CNN network...")

    # 建立Mod，并绑定网络mlp,
    mod = mx.mod.Module(symbol=cnnnet,
                        context=mx.gpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

    # 训练：使用mod.fit自动绑定和训练
    # construct a callback function to save checkpoints
    # model_prefix = 'mx_cnn'
    # checkpoint = mx.callback.do_checkpoint(model_prefix)
    # mod.fit(train_iter,num_epoch=5)

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
    for epoch in range(8):
        train_iter.reset()
        metric.reset()
        for batch in train_iter:
            mod.forward(batch, is_train=True)  # compute predictions
            mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
            mod.backward()  # compute gradients
            mod.update()  # update parameters
        print('Epoch %d, Training %s' % (epoch, metric.get()))


    # 预测
    predict_res = mod.predict(test_iter, 20)
    # 处理预测结果
    test_label = []
    for each in predict_res:
        test_label.append(int(mx.ndarray.argmax(each, 0).asscalar()))

    print(test_label[0:1])
    # print(predict_res[0:4])
    # score = mod.score(train_iter, ['acc'])
    # print("Accuracy score is %f" % (score[0][1]))

    # Write
    with open("Digital/res_cnn.csv", "w") as resFile:
        csv_writer = csv.writer(resFile)
        csv_writer.writerow(['ImageId', 'Label'])
        index = 1
        for label in test_label:
            csv_writer.writerow([index, label])
            index += 1

    return


if __name__ == '__main__':
    # 全连接训练
    # nn_train()

    # cnn训练
    cnn_train()
