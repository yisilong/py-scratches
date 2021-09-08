#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


class LinearRegression(object):
    """
    一元N次函数
    f(theta) = theta0 + theta1 * x + theta2 * x^2 + ... + thetaN * x^N
    """

    def __init__(self, train, eta=0.001, degree=2, loss_func='MSE', optimizer='SGD'):
        assert loss_func in ['MSE', 'E'], 'loss_func must be MSE or E'
        assert optimizer in ['BGD', 'SGD'], 'optimizer must be BGD or SGD'
        self._loss_func = getattr(self, loss_func)
        self._optimizer = getattr(self, optimizer)
        self.train_x = train[:, 0]
        self.train_y = train[:, 1]
        self.mu = np.mean(self.train_x)
        self.sigma = np.std(self.train_x)
        self.eta = eta
        self._theta = np.zeros(degree + 1)
        self._theta[0] = np.min(self.train_y) - 50
        self._error_diff = 1
        self._iteration_count = 1

    def standardize(self, x):
        return (x - self.mu) / self.sigma

    def to_matrix(self, x):
        tup = [x ** i for i in range(len(self._theta))]
        return np.vstack(tup).T

    # 预测函数
    def f(self, x):
        return np.dot(x, self._theta)

    # 方差
    def E(self, x, y):
        return (1.0 / 2) * np.sum((y - self.f(x)) ** 2)

    # 均方误差
    def MSE(self, x, y):
        return (1.0 / x.shape[0]) * np.sum((y - self.f(x)) ** 2)

    # 梯度下降法
    def BGD(self, X):
        self._theta -= self.eta * np.dot(self.f(X) - self.train_y, X)

    # 随机梯度下降法
    def SGD(self, X):
        # 为了调整训练数据的顺序，准备随机的序列
        p = np.random.permutation(X.shape[0])
        # 随机取出训练数据，使用随机梯度下降法更新参数
        for x, y in zip(X[p, :], self.train_y[p]):
            self._theta -= self.eta * np.dot((self.f(x) - y), x)

    def draw(self, figure, ax):
        train_z = self.standardize(self.train_x)
        ax.plot(train_z, self.train_y, 'o')

        for _ in self.step(train_z):
            pass

        xs = np.linspace(-3, 3, 100)
        ys = self.f(self.to_matrix(xs))
        ax.plot(xs, ys)
        text = f'eta:{self.eta}, degree:{len(self._theta)}, loss_func:{self._loss_func.__name__}, optimizer:{self._optimizer.__name__}'
        ax.set_xlabel(text)

    def step(self, train_z):
        X = self.to_matrix(train_z)

        self._error_diff, self._iteration_count = 1, 1
        error = self._loss_func(X, self.train_y)
        while self._error_diff > 0.01:
            self._optimizer(X)
            curr_error = self._loss_func(X, self.train_y)
            self._error_diff = error - curr_error
            # print(f'第{self._iteration_count}次, theta:{self._theta}, 差值:{self._error_diff:.4f}')
            error = curr_error
            self._iteration_count += 1
            yield object()

    def draw_animation(self, figure, ax):
        train_z = self.standardize(self.train_x)
        ax.plot(train_z, self.train_y, 'o')

        xs = np.linspace(-3, 3, 100)
        zx = self.to_matrix(xs)
        ys = self.f(zx)
        line, = ax.plot(xs, ys)

        def animate(i):
            ys = self.f(zx)
            line.set_ydata(ys)
            text = f'count:{self._iteration_count}, theta:{self._theta}, loss:{self._error_diff:.4f}'
            ax.set_xlabel(text)
            return line,

        def init():
            line.set_ydata(ys)
            return line,

        ani = animation.FuncAnimation(fig=figure,
                                      func=animate,
                                      frames=self.step(train_z),
                                      init_func=init,
                                      interval=1,
                                      blit=False,
                                      repeat=False)
        return ani


if __name__ == '__main__':
    train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all', sharex='all', figsize=(15, 7))
    ml1 = LinearRegression(train, degree=2)
    ml1.draw(fig, ax1)
    ml2 = LinearRegression(train, degree=2)
    _ = ml2.draw_animation(fig, ax2)
    plt.show()
