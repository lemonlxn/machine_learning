# /usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/3/31 17:03
# @Author  : lemon

import numpy as np
import pandas as pd


def fill_missing_rf(X, y, to_fill_col):
    '''
    X : 完整的特征
    y : 完整的标签
    to_fill_col : 需要填补缺失值的列名
    '''

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    df = pd.concat([X.loc[:, X.columns != to_fill_col], pd.DataFrame(y)], axis=1)
    to_fill = X.loc[:, to_fill_col]

    y_train = to_fill[to_fill.notnull()]
    X_train = df.iloc[y_train.index, :]

    y_test = to_fill[to_fill.isnull()]
    X_test = df.iloc[y_test.index, :]

    rf_clf = RandomForestRegressor(n_estimators=100)
    rf_clf.fit(X_train, y_train)
    y_predict = rf_clf.predict(X_test)

    return y_predict