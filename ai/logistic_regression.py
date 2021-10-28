#!/usr/bin/env python3
# coding: utf-8

import util
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


class LogisticRegression(object):
    """
    逻辑回归
    theta0 + theta1 * x + theta2 * x^2 + ... + thetaN * x^N + theta * z = 0
    """

    def __init__(self, eta=0.005, loss='MSE', optimizer='SGD', regularization='L2'):
        assert optimizer in ['BGD', 'SGD'], 'optimizer must be BGD or SGD'
        self.loss_func = util.loss_func(loss)
        self.optimizer_func = getattr(self, optimizer)
        self.regularization_func = util.regularization_func(regularization)
        self.eta = eta
        self.iteration_count = 0
        self._lambda = 0.01
        self.accuracy = 0

    @property
    def theta(self):
        return self._theta

    @property
    def error(self):
        return 1 - self.accuracy

    # 分类函数
    def f(self, x):
        return util.sigmoid(np.dot(x, self._theta))

    # 梯度下降法
    def BGD(self, X, y):
        self._theta -= self.eta * (np.dot(self.f(X) - y, X) + self.regularization_func())

    # 随机梯度下降法
    def SGD(self, X, y):
        # 为了调整训练数据的顺序，准备随机的序列
        p = np.random.permutation(X.shape[0])
        # 随机取出训练数据，使用随机梯度下降法更新参数
        for x, y in zip(X[p, :], y[p]):
            self.BGD(x, y)

    # 训练
    def fit(self, X, y):
        for _ in self.step(X, y):
            pass
        return self

    # 预测
    def predict(self, x):
        x = util.standardize(x)
        return self.classify(x)

    def classify(self, x):
        return (self.f(x) >= 0.5).astype(np.int)

    def step(self, X, y):
        self._theta = np.random.rand(X.shape[-1])
        while self.accuracy < 0.995:
            self.optimizer_func(X, y)
            self.accuracy = len(X[self.classify(X) == y]) / len(X)
            self.iteration_count += 1
            # print(f'count:{self.iteration_count}, theta:{self._theta}, accuracy:{self.accuracy}')
            yield object()


def draw(model, X, y, figure, ax, degree=1):
    i_o, i_x = y == 1, y == 0
    ax.plot(X[i_o, 0], X[i_o, 1], 'o')
    ax.plot(X[i_x, 0], X[i_x, 1], 'x')

    indexes = np.argsort(X[:, 0], axis=0)

    Z = X[:, -1:]
    z_offset = np.average(Z, axis=0)
    Z = Z - z_offset

    X1 = util.standardize(X[:, :-1])
    X1 = util.to_matrix(X1, degree)

    model.fit(np.hstack((X1, Z)), y)

    z = -np.dot(X1, model.theta[:-1]) / model.theta[-1]
    ax.plot(X[indexes, 0], z[indexes] + z_offset)

    optimizer_func_name = model.optimizer_func.__name__
    text = f'eta:{model.eta}, degree:{degree}, optimizer:{optimizer_func_name}'
    ax.set_xlabel(text)


def draw_animation(model, X, y, figure, ax, degree=1):
    i_o, i_x = y == 1, y == 0
    ax.plot(X[i_o, 0], X[i_o, 1], 'o')
    ax.plot(X[i_x, 0], X[i_x, 1], 'x')

    indexes = np.argsort(X[:, 0], axis=0)

    Z = X[:, -1:]
    line, = ax.plot(X[indexes, 0], np.full(X.shape[0], np.min(Z)))
    z_offset = np.average(Z, axis=0)
    Z -= z_offset

    X1 = util.standardize(X[:, :-1])
    X1 = util.to_matrix(X1, degree)

    def animate(i):
        z = -np.dot(X1, model.theta[:-1]) / model.theta[-1]
        line.set_ydata(z[indexes] + z_offset)
        text = f'count:{model.iteration_count}, theta:{model.theta}, accuracy:{model.accuracy}'
        ax.set_xlabel(text)
        return line,

    ani = animation.FuncAnimation(fig=figure,
                                  func=animate,
                                  frames=model.step(np.hstack((X1, Z)), y),
                                  interval=1,
                                  blit=False,
                                  repeat=False)
    return ani


if __name__ == '__main__':
    fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2, sharey='row', sharex='row', figsize=(13, 8))

    train = np.loadtxt('images2.csv', delimiter=',', skiprows=1)
    x, y = train[:, 0:2], train[:, 2]
    draw(LogisticRegression(), x, y, fig, ax00, 1)
    ani1 = draw_animation(LogisticRegression(), x, y, fig, ax01, 1)

    train = np.loadtxt('images3.csv', delimiter=',', skiprows=1)
    x, y = train[:, 0:2], train[:, 2]
    draw(LogisticRegression(), x, y, fig, ax10, 2)
    ani2 = draw_animation(LogisticRegression(), x, y, fig, ax11, 2)
    plt.show()
