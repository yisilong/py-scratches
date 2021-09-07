#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


class MLRegression(object):
    """
    二次函数
    f(theta) = theta0 + theta1 * x + theta2 * x * x
    """

    def __init__(self, file_name, eta=0.001, delimiter=',', skiprows=1):
        train = np.loadtxt(file_name, delimiter=delimiter, skiprows=skiprows)
        self.train_x = train[:, 0]
        self.train_y = train[:, 1]
        self.mu = np.mean(self.train_x)
        self.sigma = np.std(self.train_x)
        self.eta = eta
        self._theta = np.zeros(3)
        self._theta[0] = np.min(self.train_y) - 50
        self._error_diff = 1
        self._iteration_count = 1

    def standardize(self, x):
        return (x - self.mu) / self.sigma

    def to_matrix(self, x):
        return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

    # 预测函数
    def f(self, x):
        return np.dot(x, self._theta)

    # 目标函数
    def E(self, x, y):
        return (1.0 / 2) * np.sum((y - self.f(x)) ** 2)

    def draw(self, figure, ax):
        train_z = self.standardize(self.train_x)
        ax.plot(train_z, self.train_y, 'o')

        for _ in self.step(train_z):
            pass

        xs = np.linspace(-3, 3, 100)
        ys = self.f(self.to_matrix(xs))
        ax.plot(xs, ys)

    def step(self, train_z):
        X = self.to_matrix(train_z)

        self._error_diff, self._iteration_count = 1, 1
        error = self.E(X, self.train_y)
        while self._error_diff > 0.01:
            self._theta = self._theta - self.eta * np.dot(self.f(X) - self.train_y, X)
            curr_error = self.E(X, self.train_y)
            self._error_diff = error - curr_error
            error = curr_error
            # print(f'第{self._iteration_count}次, theta:{self._theta}, 差值:{self._error_diff:.4f}')
            self._iteration_count += 1
            yield object()

    def draw_animation(self, figure, ax):
        train_z = self.standardize(self.train_x)
        ax.plot(train_z, self.train_y, 'o')

        xs = np.linspace(-3, 3, 100)
        ys = self.f(self.to_matrix(xs))
        line, = ax.plot(xs, ys)

        def animate(i):
            ys = self.f(self.to_matrix(xs))
            line.set_ydata(ys)
            text = f'count:{self._iteration_count}, theta:{self._theta}, diff:{self._error_diff:.4f}'
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
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all', sharex='all', figsize=(15, 7))
    ml1 = MLRegression('click.csv')
    ml1.draw(fig, ax1)
    ml2 = MLRegression('click.csv')
    _ = ml2.draw_animation(fig, ax2)
    plt.show()
