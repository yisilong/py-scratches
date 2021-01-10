#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# f(theta) = theta0 + theta1 * x

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
theta0 = np.random.rand()
theta1 = np.random.rand()


# 预测函数
def f(x):
    return theta0 + theta1 * x


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
error = E(train_z, train_y)
while diff > 0.01:
    new_theta0 = theta0 - ETA * np.sum(f(train_z) - train_y)
    new_theta1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    theta0, theta1 = new_theta0, new_theta1
    curr_error = E(train_z, train_y)
    diff = error - curr_error
    error = curr_error
    count += 1
    print(f'第{count}次， theta0:{theta0:.3f}, theta1:{theta1:.3f}, 差值:{diff:.4f}')

xs = np.linspace(-3, 3, 100)
ys = f(xs)
plt.plot(xs, ys)

plt.show()
