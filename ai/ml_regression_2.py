#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# f(theta) = theta0 + theta1 * x + theta2 * x^2

# 数据图像
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

# 数据标准化/z-score规范化
mu = np.mean(train_x)
sigma = np.std(train_x)


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)

plt.plot(train_z, train_y, 'o')

# 训练
theta = np.random.rand(3)


def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T


X = to_matrix(train_z)


# 预测函数
def f(x):
    return np.dot(x, theta)


# 目标函数
def E(x, y):
    return (1.0 / 2) * np.sum((y - f(x)) ** 2)


# 学习率
ETA = 0.001
# 误差的差值
diff = 1
# 更新次数
count = 1
# 重复学习
error = E(X, train_y)
while diff > 0.01:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    curr_error = E(X, train_y)
    diff = error - curr_error
    error = curr_error
    print(f'第{count}次， theta:{theta}, 差值:{diff:.4f}')
    count += 1

xs = np.linspace(-3, 3, 100)
ys = f(to_matrix(xs))
plt.plot(xs, ys)

plt.show()
