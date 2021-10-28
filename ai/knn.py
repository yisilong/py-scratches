#!/usr/bin/env python3
# coding: utf-8

import util
import numpy as np


class KNN(object):
    """
    1. 计算已知类别数据集中的点与当前点之间的距离；
    2. 按照距离递增次序排序；
    3. 选取与当前点距离最小的k个点；
    4. 确定前k个点所在类别的出现频率；
    5. 返回前k个点出现频率最高的类别作为当前点的预测分类
    """

    def __init__(self, k=10):
        self.k = k

    def get_neighbors(self, X, target):
        """
        计算最近的k个近邻
        """
        distances = util.euclidean_distance(target, X)
        print(distances)
        indexes = np.argsort(distances)
        return indexes[:self.k]

    def fit(self, X, y):
        """
        训练
        """
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        """
        预测
        """
        assert hasattr(self, '_X')
        assert hasattr(self, '_y')

        y_predicts = []
        for one in X:
            top_k = self.get_neighbors(self._X, one)
            kind = np.argmax(np.bincount(self._y[top_k]))
            y_predicts.append(kind)

        return np.array(y_predicts)

    def score(self, X, y):
        y_predicts = self.predict(X)
        count = np.sum(y_predicts == y)
        return count / X.shape[0]


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = KNN(k=3).fit(x_train, y_train)
    # y_predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    print(score)
