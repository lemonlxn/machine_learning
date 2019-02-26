# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/2/26 13:45
# @Author  : lemon

import numpy as np
from metrics import r2_score



class LinearRegression():

    def __init__(self):
        '''
        初始化 相关系数
        '''

        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self,X_train,y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "训练数据集的样本数，需要和训练数据集的标签数一致"

        X_b = np.hstack([np.ones(shape=(len(X_train),1)),X_train])

        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self,X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
            "预测测试数据前，必须要进行predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "测试数据样本个数，需要与 _theta 参数数量保持一致"

        X_b = np.hstack([np.ones(shape=(len(X_predict), 1)), X_predict])

        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)

        return r2_score(y_test,y_predict)


    def __repr__(self):
        return 'LinearRegression()'