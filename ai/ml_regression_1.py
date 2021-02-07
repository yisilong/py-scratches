#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


class MLRegression(object):
    """
    一次函数
    f(theta) = theta0 + theta1 * x
    """

    def __init__(self, file_name, eta=0.001, delimiter=',', skiprows=1):
        self.eta = eta
        train = np.loadtxt(file_name, delimiter=delimiter, skiprows=skiprows)
        self.train_x = train[:, 0]
        self.train_y = train[:, 1]
        self.mu = np.mean(self.train_x)
        self.sigma = np.std(self.train_x)
        self._theta0 = np.min(self.train_y) - 50
        self._theta1 = 0

    def standardize(self, x):
        return (x - self.mu) / self.sigma

    # 预测函数
    def f(self, x):
        return self._theta0 + self._theta1 * x

    # 目标函数
    def E(self, x, y):
        return (1.0 / 2) * np.sum((y - self.f(x)) ** 2)

    def draw(self, figure, ax):
        train_z = self.standardize(self.train_x)
        ax.plot(train_z, self.train_y, 'o')

        for _ in self.step(train_z):
            pass

        xs = np.linspace(-3, 3, 100)
        ys = self.f(xs)
        ax.plot(xs, ys)

    def step(self, train_z):
        diff, count = 1, 1
        error = self.E(train_z, self.train_y)
        while diff > 0.01:
            new_theta0 = self._theta0 - self.eta * np.sum(self.f(train_z) - self.train_y)
            new_theta1 = self._theta1 - self.eta * np.sum((self.f(train_z) - self.train_y) * train_z)
            self._theta0, self._theta1 = new_theta0, new_theta1
            curr_error = self.E(train_z, self.train_y)
            diff = error - curr_error
            print(f'第{count}次， theta0:{self._theta0:.3f}, theta1:{self._theta1:.3f}, 差值:{diff:.4f}')
            error = curr_error
            count += 1
            yield object()

    def draw_animation(self, figure, ax):
        train_z = self.standardize(self.train_x)
        ax.plot(train_z, self.train_y, 'o')

        xs = np.linspace(-3, 3, 100)
        ys = self.f(xs)
        line, = ax.plot(xs, ys)

        def animate(i):
            ys = self.f(xs)
            line.set_ydata(ys)
            return line,

        def init():
            line.set_ydata(ys)
            return line,

        ani = animation.FuncAnimation(fig=figure,
                                      func=animate,
                                      frames=self.step(train_z),
                                      init_func=init,
                                      interval=5,
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
