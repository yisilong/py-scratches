#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


class Perceptron(object):
    """
    感知机分类
    """

    def __init__(self, epoch=10):
        self._theta = np.array([1, 0], dtype=float)
        self._epoch = epoch
        self._iteration_count = 1

    # 预测函数
    def predict(self, x):
        return np.dot(x, self._theta)

    # 判别函数
    def _predict(self, x):
        if np.dot(x, self._theta) > 0:
            return 1
        return -1

    def fit(self, X, y):
        for _ in self.step(X, y):
            pass

        return self

    def step(self, X, y):
        for i in range(self._epoch):
            for x, y_ in zip(X, y):
                if self._predict(x) != y_:
                    self._theta += y_ * x
                    # print(f'第{i + 1}次, theta:{self._theta}')
                self._iteration_count += 1
                yield object()

    def draw(self, X, y, figure, ax):
        i_o, i_x = y == 1, y == -1
        ax.plot(X[i_o, 0], X[i_o, 1], 'o')
        ax.plot(X[i_x, 0], X[i_x, 1], 'x')

        self.fit(X, y)

        indexes = np.argsort(X, axis=0)[:, 0]
        x0 = X[indexes, 0]
        x1 = -self._theta[0] * x0 / self._theta[1]
        ax.plot(x0, x1)

    def draw_animation(self, X, y, figure, ax):
        i_o, i_x = y == 1, y == -1
        ax.plot(X[i_o, 0], X[i_o, 1], 'o')
        ax.plot(X[i_x, 0], X[i_x, 1], 'x')

        indexes = np.argsort(X, axis=0)[:, 0]
        x0 = X[indexes, 0]
        x1 = -self._theta[0] * x0 / self._theta[1]
        line, = ax.plot(x0, x1)

        def animate(i):
            y1 = -self._theta[0] * x0 / self._theta[1]
            line.set_ydata(y1)
            text = f'count:{self._iteration_count}, theta:{self._theta}'
            ax.set_xlabel(text)
            return line,

        ani = animation.FuncAnimation(fig=figure,
                                      func=animate,
                                      frames=self.step(X, y),
                                      interval=10,
                                      blit=False,
                                      repeat=False)
        return ani


if __name__ == '__main__':
    train = np.loadtxt('images1.csv', delimiter=',', skiprows=1)
    x, y = train[:, 0:2], train[:, 2]

    fig, (ax0, ax1) = plt.subplots(1, 2, sharey='all', sharex='all', figsize=(13, 8))
    Perceptron().draw(x, y, fig, ax0)
    ani = Perceptron().draw_animation(x, y, fig, ax1)
    plt.show()
