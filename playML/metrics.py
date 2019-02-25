# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/1/29 17:58
# @Author  : lemon

import numpy as np
from math import sqrt


def accuracy_score(predict_label,y_test_label):
    '''
    :param predict:预测数据集
    :param test:训练测试数据集
    :return:
    '''

    return np.sum(predict_label == y_test_label) / len(y_test_label)


def mean_squared_error(y_true,y_predict):
    '''

    :param y_true: 真实数据
    :param y_predict: 预测数据
    :return: 返回 MSE (mean squared error)
    '''
    assert len(y_true) == len(y_predict),\
        '真实数据和预测数据数量应一致'

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true,y_predict):
    '''
    计算 y_true,y_predict 之间的 RMSE
    '''
    return sqrt(mean_squared_error(y_true,y_predict))


def mean_absolute_error(y_true,y_predict):
    '''
    计算 y_true,y_predict 之间的 MAE
    '''
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)