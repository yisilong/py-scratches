#!/usr/bin/env python3
# coding: utf-8
import array

import util
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


class LinearRegression(object):
    """
    一元N次函数
    f(theta) = theta0 + theta1 * x + theta2 * x^2 + ... + thetaN * x^N
    """

    def __init__(self, eta=0.001, loss_func='MSE', optimizer='SGD', regularization='L2'):
        assert loss_func in ['MSE', 'E'], 'loss_func must be MSE or E'
        assert optimizer in ['BGD', 'SGD'], 'optimizer must be BGD or SGD'
        assert regularization in ['L1', 'L2'], 'regularization must be L1 or L2'
        self.loss_func = getattr(self, loss_func)
        self.optimizer_func = getattr(self, optimizer)
        self.regularization_func = getattr(self, regularization)
        self.eta = eta
        self._error = np.NAN
        self.iteration_count = 1
        self._lambda = 0.01

    @property
    def theta(self):
        return self._theta

    @property
    def error(self):
        return self._error

    # 拟合函数
    def f(self, x):
        return np.dot(x, self._theta)

    # 方差
    def E(self, x, y):
        return (1.0 / 2) * np.sum((y - self.f(x)) ** 2)

    # 均方误差
    def MSE(self, x, y):
        return (1.0 / x.shape[0]) * np.sum((y - self.f(x)) ** 2)

    # L1正则化
    def L1(self):
        l1 = self._lambda * np.array(list(map(lambda _: 1 if _ > 0 else -1, self._theta)))
        l1[0] = 0
        return l1

    # L2正则化
    def L2(self):
        l2 = self._lambda * self._theta
        l2[0] = 0
        return l2

    # 梯度下降法
    def BGD(self, X, y):
        self._theta -= self.eta * (np.dot(self.f(X) - y, X) + self.regularization_func())

    # 随机梯度下降法
    def SGD(self, X, y):
        # 为了调整训练数据的顺序，准备随机的序列
        p = np.random.permutation(X.shape[0])
        # 随机取出训练数据，使用随机梯度下降法更新参数
        for x, y in zip(X[p, :], y[p]):
            self._theta -= self.eta * (np.dot((self.f(x) - y), x) + self.regularization_func())

    # 训练
    def fit(self, X, y):
        X = util.standardize(X, axis=0)
        for _ in self.step(X, y):
            pass
        return self

    # 预测
    def predict(self, x):
        x = util.standardize(x, axis=0)
        return self.f(x)

    def step(self, X, y):
        self._theta = np.random.random(X.shape[-1])
        error_diff = 1
        self._error = self.loss_func(X, y)
        while error_diff > 0.01:
            self.optimizer_func(X, y)
            curr_error = self.loss_func(X, y)
            error_diff = abs(self._error - curr_error)
            self.iteration_count += 1
            self._error = curr_error
            # print(f'count:{self.iteration_count}, theta:{self._theta}, loss:{self._error:.4f}')
            yield object()


def draw(model, X, y, figure, ax, degree=1):
    ax.scatter(X, y)

    X1 = util.to_matrix(X, degree)
    y_predict = model.fit(X1, y).predict(X1)

    indexes = np.argsort(X, axis=0)[:, 0]
    ax.plot(X[indexes], y_predict[indexes], color='red')

    loss_func_ = model.loss_func.__name__
    optimizer_func_ = model.optimizer_func.__name__
    text = f'eta:{model.eta}, degree:{degree}, loss:{loss_func_}, optimizer:{optimizer_func_}'
    ax.set_xlabel(text)


def draw_animation(model, X, y, figure, ax, degree=1):
    ax.scatter(X, y)

    indexes = np.argsort(X, axis=0)[:, 0]
    y_predict = np.zeros(X.shape[0])
    line, = ax.plot(X[indexes], y_predict[indexes], color='red')

    X = util.to_matrix(X, degree)
    X = util.standardize(X, axis=0)

    def animate(i):
        y_predict = model.f(X)
        line.set_ydata(y_predict[indexes])
        text = f'count:{model.iteration_count}, theta:{model.theta}, loss:{model.error:.4f}'
        ax.set_xlabel(text)
        return line,

    ani = animation.FuncAnimation(fig=figure,
                                  func=animate,
                                  frames=model.step(X, y),
                                  interval=1,
                                  blit=False,
                                  repeat=False)
    return ani


if __name__ == '__main__':
    train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
    x, y = train[:, 0].reshape(-1, 1), train[:, 1]
    fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2, sharey='all', sharex='all', figsize=(13, 8))
    draw(LinearRegression(), x, y, fig, ax00, 1)
    ani1 = draw_animation(LinearRegression(), x, y, fig, ax01, 1)
    draw(LinearRegression(), x, y, fig, ax10, 2)
    ani2 = draw_animation(LinearRegression(), x, y, fig, ax11, 2)
    plt.show()
