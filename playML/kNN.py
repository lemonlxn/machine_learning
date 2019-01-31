# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/1/28 21:16
# @Author  : lemon

import numpy as np
from math import sqrt
from collections import Counter

from .metrics import accuracy_score

class KNNClassifier():
    '''
    make comments

    X_train        测试数据集
    y_train_label  测试数据标签集

    X_test         训练数据集
    y_test_label   训练数据标签集

    '''
    def __init__(self,k):
        assert k>=1,'k 必须是整数，且得大于等于1'
        self.k = k
        self._X_train = None
        self._y_train_label = None

    def fit(self,X_train,y_train_label):

        assert self.k <= X_train.shape[0],\
            'k的数量，需要小于等于训练集的样本数量'
        assert X_train.shape[0] == y_train_label.shape[0],\
            '训练集的样本数量，必须和训练集标签数量一致'

        self._X_train = X_train
        self._y_train_label = y_train_label

        return self

    def predict(self,X_predict):
        assert self._X_train is not None and self._y_train_label is not None,\
            '训练集样本，以及标签，不能为空'
        assert X_predict.shape[1] == self._X_train.shape[1],\
            '预测的样本列数，必须要和训练集列数保持一致'

        predict_label = [self._predict(x) for x in X_predict]
        return np.array(predict_label)

    def _predict(self,x):

        distinces = [ sqrt(np.sum((i-x) **2))  for i in self._X_train]
        nearest_distance = np.argsort(distinces)

        top_nearest_label = [self._y_train_label[i] for i in nearest_distance[:self.k]]
        label_type  = Counter(top_nearest_label)

        predict_label = label_type.most_common(1)[0][0]

        return predict_label



    def score(self,X_test,y_test_label):

        predict_label = self.predict(X_test)

        return accuracy_score(predict_label,y_test_label)

    def __repr__(self):
        return 'k = %d' % self.k




