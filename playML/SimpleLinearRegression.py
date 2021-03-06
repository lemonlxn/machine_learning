# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/2/4 18:16
# @Author  : lemon

import numpy as np
from metrics import r2_score


class SimpleLinearRegression1():
    '''
    该这个版本是初步的。
    简单线性回归测试，以下面 SimpleLinearRegression 这个类为准

    '''
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
        '''
        通过训练数据集，找到最合适的 a、b参数
        '''
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
        '''
        经过fit后，得到a、b参数。
        通过传入预测数据集，返回相应的预测结果
        '''
        assert isinstance(x_predict,np.ndarray),\
            '传入的数据集，只能是一个numpy.ndarray 类型'
        assert x_predict.ndim == 1,\
            '暂时只处理一维数组线性回归'
        assert self.a_ is not None and self.b_ is not None,\
            '必须先fit数据，得到参数a,b后，才能预测数据'
        return np.array([self._predict(i) for i in x_predict])

    def _predict(self,x):
        return self.a_ * x + self.b_

    def score(self,x_test,y_test):
        # 根据测试数据，与预测数据拟合情况，返回相应评分

        y_predict = self.predict(x_test)

        return r2_score(y_test,y_predict)

    def __repr__(self):
        return 'SimpleLinearRegression'







