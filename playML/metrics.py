# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/1/29 17:58
# @Author  : lemon

import numpy as np
from math import sqrt


def accuracy_score(y_predict,y_test):
    '''
    :param y_predict:预测数据集
    :param y_test:测试数据集
    :return: 计算模型拟合度
    '''
    assert len(y_predict) == len(y_test),\
        '预测数据集，需要和测试数据集数量保持一致'

    return np.sum(y_predict == y_test) / len(y_test)


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
    assert len(y_true) == len(y_predict), \
        '预测数据集，需要和测试数据集数量保持一致'

    return sqrt(mean_squared_error(y_true,y_predict))


def mean_absolute_error(y_true,y_predict):
    '''
    计算 y_true,y_predict 之间的 MAE
    '''
    assert len(y_true) == len(y_predict), \
        '预测数据集，需要和测试数据集数量保持一致'

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)



def r2_score(y_true,y_predict):
    '''
    :param y_true: 真实数据
    :param y_predict: 预测数据
    :return: 返回 R^2 模型拟合情况。即 1 - MSE(y_true,y_predict) / Var(y_true)
    '''
    assert len(y_true) == len(y_predict),\
        '预测数据集，需要和测试数据集数量保持一致'

    return 1 - mean_squared_error(y_true,y_predict) / np.var(y_true)