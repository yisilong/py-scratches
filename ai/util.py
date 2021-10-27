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
