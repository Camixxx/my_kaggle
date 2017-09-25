import csv
import mxnet as mx
import logging
import argparse
PATH = "C:/Users/CamiXXX/Desktop/code/kaggel/Digital/"
def nn_train():
    """ MLP network architecture for MNIST
        全链接神经网络识别手写体 """
    train_csv = csv.reader(open(PATH+"train.csv", encoding='utf-8'))
    for row in train_csv:
        print(row)

    # mx.io.MXDataIter()
    # 网络搭建


if __name__ == '__main__':
    nn_train()