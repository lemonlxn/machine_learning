# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/1/29 17:58
# @Author  : lemon

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


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
    :param theta:   np.random.randn(X_b.shape[1])
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


def PolynomialRegression(degree):
    '''

    :param degree: 多项式阶数
    :return: 返回一个数据规整后的多项式线性回归
    '''
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('stand', StandardScaler()),
        ('lin_reg', LinearRegression())
    ])


def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    '''
    :param algo:    线性回归机器学习函数 如 LinearRegression() 、 PolynomialRegression(degree=20)
    :param X_train: 训练数据集
    :param X_test:  测试数据集
    :param y_train: 训练数据集标注
    :param y_test:  测试数据集标注
    :return:        展示 线性回归机器学习函数，训练数据集 与 测试数据集 大致学习误差曲线
    '''
    from sklearn.metrics import mean_squared_error

    train_score = []
    test_score = []

    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        y_test_predict = algo.predict(X_test)

        train_score.append(mean_squared_error(y_train[:i], y_train_predict))
        test_score.append(mean_squared_error(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train) + 1)], np.sqrt(train_score), label='train')
    plt.plot([i for i in range(1, len(X_train) + 1)], np.sqrt(test_score), label='test')
    plt.legend()
    plt.axis([0, len(X_train) + 1, 0, 4])
    plt.show()


def plot_decision_boundary(model, axis):
    '''
    绘制决策边界
    对于逻辑回归，暂时只处理二分类问题
    对于KNN算法，可处理多分类问题

    :param model: 逻辑回归、KNN分类实例化之后的对象 如 log_reg = LogisticRegression()
                                                  log_reg.fit(iris.data[:,:2],iris.target)
                                                  然后将 log_reg 传入
    :param axis:  坐标轴范围，如 axis=[4, 8, 1.5, 4.5]
    :return:

     '''


    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)