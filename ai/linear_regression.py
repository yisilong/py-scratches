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

    def __init__(self, eta=0.001, degree=2, loss_func='MSE', optimizer='SGD'):
        assert loss_func in ['MSE', 'E'], 'loss_func must be MSE or E'
        assert optimizer in ['BGD', 'SGD'], 'optimizer must be BGD or SGD'
        self._loss_func = getattr(self, loss_func)
        self._optimizer = getattr(self, optimizer)
        self.eta = eta
        self.degree = degree
        self._theta = np.zeros(degree + 1)
        # self._theta[0] = np.min(self.train_y) - 50
        self._error_diff = 1
        self._iteration_count = 1

    # 标准化(归一化)
    def standardize(self, x):
        mu = np.mean(x)
        sigma = np.std(x)
        return (x - mu) / sigma

    def to_matrix(self, x):
        tup = [x ** i for i in range(len(self._theta))]
        return np.vstack(tup).T

    # 预测函数
    def predict(self, x):
        x = self.standardize(x)
        x = self.to_matrix(x)
        return np.dot(x, self._theta)

    # 预测函数
    def _predict(self, x):
        return np.dot(x, self._theta)

    # 方差
    def E(self, x, y):
        return (1.0 / 2) * np.sum((y - self._predict(x)) ** 2)

    # 均方误差
    def MSE(self, x, y):
        return (1.0 / x.shape[0]) * np.sum((y - self._predict(x)) ** 2)

    # 梯度下降法
    def BGD(self, X, train_y):
        self._theta -= self.eta * np.dot(self._predict(X) - train_y, X)

    # 随机梯度下降法
    def SGD(self, X, train_y):
        # 为了调整训练数据的顺序，准备随机的序列
        p = np.random.permutation(X.shape[0])
        # 随机取出训练数据，使用随机梯度下降法更新参数
        for x, y in zip(X[p, :], train_y[p]):
            self._theta -= self.eta * np.dot((self._predict(x) - y), x)

    def fit(self, X, y):
        self._theta[0] = np.min(y) - 50
        X = self.standardize(X)
        X = self.to_matrix(X)

        for _ in self.step(X, y):
            pass

        return self

    def step(self, X, y):
        self._error_diff, self._iteration_count = 1, 1
        error = self._loss_func(X, y)
        while self._error_diff > 0.01:
            self._optimizer(X, y)
            curr_error = self._loss_func(X, y)
            self._error_diff = error - curr_error
            # print(f'第{self._iteration_count}次, theta:{self._theta}, 差值:{self._error_diff:.4f}')
            error = curr_error
            self._iteration_count += 1
            yield object()

    def draw(self, X, y, figure, ax):
        ax.scatter(X, y)

        y_predict = self.fit(X, y).predict(X)
        indexes = np.argsort(X)
        ax.plot(X[indexes], y_predict[indexes], color='red')

        text = f'eta:{self.eta}, degree:{self.degree}, '
        text += f'loss_func:{self._loss_func.__name__}, optimizer:{self._optimizer.__name__}'
        ax.set_xlabel(text)

    def draw(self, X, y, figure, ax):
        ax.scatter(X, y)

        y_predict = self.fit(X, y).predict(X)
        indexes = np.argsort(X)
        ax.plot(X[indexes], y_predict[indexes], color='red')

        text = f'eta:{self.eta}, degree:{self.degree}, '
        text += f'loss_func:{self._loss_func.__name__}, optimizer:{self._optimizer.__name__}'
        ax.set_xlabel(text)


    def draw_animation(self, X, y, figure, ax):
        ax.scatter(X, y)

        indexes = np.argsort(X)
        y_predict = np.zeros(X.shape[0])
        line, = ax.plot(X[indexes], y_predict[indexes], color='red')

        self._theta[0] = np.min(y) - 50
        X1 = self.standardize(X)
        X1 = self.to_matrix(X1)

        def animate(i):
            y_predict = self._predict(X1)
            line.set_ydata(y_predict[indexes])
            text = f'count:{self._iteration_count}, theta:{self._theta}, loss:{self._error_diff:.4f}'
            ax.set_xlabel(text)
            return line,

        ani = animation.FuncAnimation(fig=figure,
                                        func=animate,
                                        frames=self.step(X1, y),
                                        interval=1,
                                        blit=False,
                                        repeat=False)
        return ani


if __name__ == '__main__':
    train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
    x, y = train[:, 0], train[:, 1]
    fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2, sharey='all', sharex='all', figsize=(13, 8))
    LinearRegression(degree=1).draw(x, y, fig, ax00)
    ani1 = LinearRegression(degree=1).draw_animation(x, y, fig, ax01)
    LinearRegression(degree=2).draw(x, y, fig, ax10)
    ani2 = LinearRegression(degree=2).draw_animation(x, y, fig, ax11)
    plt.show()
