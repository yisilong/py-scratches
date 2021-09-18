#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


# 标准化(归一化)
def standardize(x, axis=None):
    mu = np.mean(x, axis=axis)
    sigma = np.std(x, axis=axis)
    return (x - mu) / sigma


def to_matrix(x, degree):
    return PolynomialFeatures(degree).fit_transform(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
