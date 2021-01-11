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
        self.eta = eta
        self._theta = np.random.rand(3)
        train = np.loadtxt(file_name, delimiter=delimiter, skiprows=skiprows)
        self.train_x = train[:, 0]
        self.train_y = train[:, 1]
        self.mu = np.mean(self.train_x)
        self.sigma = np.std(self.train_x)

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

    def draw(self):
        train_z = self.standardize(self.train_x)
        plt.plot(train_z, self.train_y, 'o')
        X = self.to_matrix(train_z)

        diff, count = 1, 1
        error = self.E(X, self.train_y)
        while diff > 0.01:
            self._theta = self._theta - self.eta * np.dot(self.f(X) - self.train_y, X)
            curr_error = self.E(X, self.train_y)
            diff = error - curr_error
            error = curr_error
            print(f'第{count}次， theta:{self._theta}, 差值:{diff:.4f}')
            count += 1

        xs = np.linspace(-3, 3, 100)
        ys = self.f(self.to_matrix(xs))
        plt.plot(xs, ys)

        plt.show()

    def step(self, train_z):
        X = self.to_matrix(train_z)

        diff, count = 1, 1
        error = self.E(X, self.train_y)
        while diff > 0.01:
            self._theta = self._theta - self.eta * np.dot(self.f(X) - self.train_y, X)
            curr_error = self.E(X, self.train_y)
            diff = error - curr_error
            error = curr_error
            print(f'第{count}次， theta:{self._theta}, 差值:{diff:.4f}')
            count += 1
            yield object()

    def draw_animation(self):
        fig, ax = plt.subplots()

        train_z = self.standardize(self.train_x)
        ax.plot(train_z, self.train_y, 'o')

        xs = np.linspace(-3, 3, 100)
        ys = self.f(self.to_matrix(xs))
        line, = ax.plot(xs, ys)

        def animate(i):
            ys = self.f(self.to_matrix(xs))
            line.set_ydata(ys)
            return line,

        def init():
            line.set_ydata(ys)
            return line,

        ani = animation.FuncAnimation(fig=fig,
                                      func=animate,
                                      frames=self.step(train_z),
                                      init_func=init,
                                      interval=20,
                                      blit=False,
                                      repeat=False)
        plt.show()


if __name__ == '__main__':
    ml = MLRegression('click.csv')
    ml.draw_animation()
