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



def dJ_debug(theta,X_b,y,epsilon = 0.01):

    '''
    :param theta:
    :param X_b:     np.hstack([np.ones((len(X), 1)), X])
    :param y:       标签值
    :param epsilon: theta 间距
    :return:        返回该梯度上对应的比较精准的 theta 值
                    不足点：由于每项theta需要计算 2次 样本数，总体运算速度较慢，不过可以作为调试，作为参照。
    '''

    def J(theta,X_b,y):
        try:
            return np.sum((y-X_b.dot(theta)) ** 2) / len(y)
        except:
            return float('inf')

    result = np.empty(len(theta))

    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon

        theta_2 = theta.copy()
        theta_2[i] -= epsilon

        result[i] = (J(theta_1,X_b,y) - J(theta_2,X_b,y)) / 2 * epsilon

    return result

