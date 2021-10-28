#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


# 标准化(归一化)
def standardize(x):
    assert x.ndim == 2
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x = x - mu
    indexes = (sigma == 0)
    if np.any(indexes):
        sigma[indexes] = 1
        x[:, indexes] = 1
    return x / sigma


def to_matrix(x, degree):
    return PolynomialFeatures(degree).fit_transform(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def l1_regularization(lambda_, coef_):
    """
    L1正则化
    """
    l1 = lambda_ * np.array(list(map(lambda _: 1 if _ > 0 else -1, coef_)))
    l1[0] = 0
    return l1


def l2_regularization(lambda_, coef_):
    """
    L2正则化
    """
    l2 = lambda_ * coef_
    l2[0] = 0
    return l2


def regularization_func(kind=None):
    assert kind in (None, 'L1', 'L2'), 'kind must be L1 or L2'
    if kind == 'L1':
        func = l1_regularization
    elif kind == 'L2':
        func = l2_regularization
    else:
        def func(*args):
            return 0
    return func


def E(X, y, f):
    """
    方差
    """
    return (1.0 / 2) * np.sum((y - f(X)) ** 2)


def MSE(X, y, f):
    """
    均方误差
    """
    return (1.0 / X.shape[0]) * np.sum((y - f(X)) ** 2)


def loss_func(kind='E'):
    assert kind in ['MSE', 'E'], 'kind must be MSE or E'
    return MSE if kind == 'MSE' else E


def euclidean_distance(vec1, vec2):
    """
    计算欧几里得距离
    """
    return np.sqrt(np.sum(np.square(vec1 - vec2), axis=-1))
