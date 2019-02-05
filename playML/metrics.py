# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/1/29 17:58
# @Author  : lemon

import numpy as np

def accuracy_score(predict_label,y_test_label):
    '''
    :param predict:预测数据集
    :param test:训练测试数据集
    :return:
    '''

    return np.sum(predict_label == y_test_label) / len(y_test_label)


