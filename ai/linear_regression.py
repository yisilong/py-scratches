#!/usr/bin/env python3
# coding: utf-8

import util
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


class LinearRegression(object):
    """
    一元N次函数
    f(theta) = theta0 + theta1 * x + theta2 * x^2 + ... + thetaN * x^N
    """

    def __init__(self, eta=0.005, loss='MSE', optimizer='SGD', regularization='L2'):
        assert optimizer in ['BGD', 'SGD'], 'optimizer must be BGD or SGD'
        self.loss_func = util.loss_func(loss)
        self.optimizer_func = getattr(self, optimizer)
        self.regularization_func = util.regularization_func(regularization)
        self.eta = eta
        self.error = np.NAN
        self.iteration_count = 1
        self._lambda = 0.01

    @property
    def theta(self):
        return self._theta

    # 拟合函数
    def f(self, x):
        return np.dot(x, self._theta)

    # 梯度下降法
    def BGD(self, X, y):
        v = self.regularization_func(self.theta)
        self._theta -= self.eta * (np.dot(self.f(X) - y, X) + v)

    # 随机梯度下降法
    def SGD(self, X, y):
        # 为了调整训练数据的顺序，准备随机的序列
        p = np.random.permutation(X.shape[0])
        # 随机取出训练数据，使用随机梯度下降法更新参数
        for x, y in zip(X[p, :], y[p]):
            self.BGD(x, y)

    # 训练
    def fit(self, X, y):
        X = util.standardize(X)

        for _ in self.step(X, y):
            pass

        return self

    # 预测
    def predict(self, X):
        X = util.standardize(X)
        return self.f(X)

    def step(self, X, y):
        self._theta = np.random.random(X.shape[-1])
        error_diff, self.iteration_count = 1, 1
        error = self.loss_func(X, y, self.f)
        while error_diff > 0.001:
            self.optimizer_func(X, y)
            self.error = self.loss_func(X, y, self.f)
            error_diff = abs(error - self.error)
            # print(f'第{self._iteration_count}次, theta:{self._theta}, 差值:{error_diff:.4f}')
            error = self.error
            self.iteration_count += 1
            yield object()


def draw(model, X, y, figure, ax, degree=1):
    ax.scatter(X, y)
    indexes = np.argsort(X[:, 0])

    X1 = util.to_matrix(X, degree)
    y_predict = model.fit(X1, y).predict(X1)

    ax.plot(X[indexes], y_predict[indexes], color='red')

    loss_func_ = model.loss_func.__name__
    optimizer_func_ = model.optimizer_func.__name__
    text = f'eta:{model.eta}, degree:{degree}, loss:{loss_func_}, optimizer:{optimizer_func_}'
    ax.set_xlabel(text)


def draw_animation(model, X, y, figure, ax, degree=1):
    ax.scatter(X, y)
    indexes = np.argsort(X[:, 0])

    y_predict = np.full(X.shape[0], np.min(y))
    line, = ax.plot(X[indexes], y_predict[indexes], color='red')

    X1 = util.to_matrix(X, degree)
    X1 = util.standardize(X1)

    def animate(i):
        y_predict = model.f(X1)
        line.set_ydata(y_predict[indexes])
        text = f'count:{model.iteration_count}, theta:{model.theta}, loss:{model.error:.4f}'
        ax.set_xlabel(text)
        return line,

    ani = animation.FuncAnimation(fig=figure,
                                  func=animate,
                                  frames=model.step(X1, y),
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
