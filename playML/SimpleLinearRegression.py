# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/2/4 18:16
# @Author  : lemon

import numpy as np


class SimpleLinearRegression1():
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        assert x_train.ndim == 1,\
            '暂时只处理一维线性回归'
        assert len(x_train) == len(y_train),\
            '训练集数据，与标签长度需要一致'

        # 分子分母初始为0
        num  = 0
        d    = 0

        # 求出一维数组均值
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        for x_i,y_i in zip(x_train,y_train):
            num += (x_i-x_mean)*(y_i-y_mean)
            d   += (x_i-x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self


    def predict(self,x_predict):
        assert isinstance(x_predict,np.ndarray),\
            '传入的数据集，只能是一个numpy.ndarray 类型'
        assert x_predict.ndim == 1,\
            '暂时只处理一维数组线性回归'
        assert self.a_ is not None and self.b_ is not None,\
            '必须先fit数据，得到参数a,b后，才能预测数据'
        return np.array([self._predict(i) for i in x_predict])

    def _predict(self,x):
        return self.a_ * x + self.b_

    def __repr__(self):
        return 'SimpleLinearRegression1'





class SimpleLinearRegression():
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        assert x_train.ndim == 1,\
            '暂时只处理一维线性回归'
        assert len(x_train) == len(y_train),\
            '训练集数据，与标签长度需要一致'

        # 求出一维数组均值
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)


        num = (x_train - x_mean).dot(y_train - y_mean)
        d   = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self


    def predict(self,x_predict):
        assert isinstance(x_predict,np.ndarray),\
            '传入的数据集，只能是一个numpy.ndarray 类型'
        assert x_predict.ndim == 1,\
            '暂时只处理一维数组线性回归'
        assert self.a_ is not None and self.b_ is not None,\
            '必须先fit数据，得到参数a,b后，才能预测数据'
        return np.array([self._predict(i) for i in x_predict])

    def _predict(self,x):
        return self.a_ * x + self.b_

    def __repr__(self):
        return 'SimpleLinearRegression'







