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


# 均方误差
def MSE(x, y):
    return (1.0 / x.shape[0]) * np.sum((y - f(x)) ** 2)


# 学习率
ETA = 0.001
# 误差的差值
diff = 1
# 更新次数
count = 1
# 重复学习
errors = [MSE(X, train_y)]
while diff > 0.01:
    # 为了调整训练数据的顺序，准备随机的序列
    p = np.random.permutation(X.shape[0])
    # 随机取出训练数据，使用随机梯度下降法更新参数
    for x, y in zip(X[p, :], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]
    print(f'第{count}次， theta:{theta}, 差值:{diff:.4f}')
    count += 1

xs = np.linspace(-3, 3, 100)
ys = to_matrix(xs)
plt.plot(xs, f(to_matrix(xs)))

plt.show()
