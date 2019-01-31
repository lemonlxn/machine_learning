# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/1/29 16:43
# @Author  : lemon

import numpy as np
from sklearn import datasets

def get_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    return X,y


def train_test_split(X,y,test_rate=0.2,seed=None):
    """
    :param X: 数据样本集
    :param y: 数据样本集对应标签
    :param test_rate: 测试数据集比例
    :param seed: np.random.seed
    :return: X_train X_test y_train y_test
   """

    assert X.shape[0] == y.shape[0],\
        '样本数量，需要和标签数保持一致'
    assert 0 <=test_rate <=1,\
        '测试数据比例在 0-1 之间'

    if seed:np.random.seed(seed)

    # 返回数据集随机索引
    shuffle_index = np.random.permutation(len(X))

    # 设定训练数据集，以及测试数据集数量
    test_size  = int(len(X) * test_rate)

    # 设定训练数据集，以及测试数据集索引
    test_index  = shuffle_index[:test_size]
    train_index = shuffle_index[test_size:]

    # 设定训练数据集，以及测试数据集样本
    X_train = X[train_index]
    y_train = y[train_index]

    X_test  = X[test_index]
    y_test  = y[test_index]

    return X_train,X_test,y_train,y_test


if __name__ == '__main__':
    X,y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    pass






