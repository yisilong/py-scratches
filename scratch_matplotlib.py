import numpy as np
import matplotlib.pyplot as plt


# Scatter 散点图
def draw_scatter():
    X = np.random.normal(size=1024)
    Y = np.random.normal(size=1024)
    T = np.arctan2(X, Y)

    plt.scatter(X, Y, s=90, c=T, alpha=0.5)
    plt.xticks(())
    plt.yticks(())
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.show()


# Bar 柱状图
def draw_bar():
    n = 12
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

    plt.xlim(-1, n)
    plt.xticks(())
    plt.ylim(-1.25, 1.25)
    plt.yticks(())

    for x, y in zip(X, Y1):
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    for x, y in zip(X, Y2):
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(x, -y - 0.05, '%.2f' % y, ha='center', va='top')

    plt.show()


# Contours 等高线图
def draw_contours():
    def f(x, y):
        # the height function
        return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

    n = 256
    X = np.linspace(-3, 3, n)
    Y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(X, Y)

    # use plt.contourf to filling contours
    # X, Y and value for (X,Y) point
    plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.hot())
    # use plt.contour to add contour lines
    C = plt.contour(X, Y, f(X, Y), 8, colors='black')
    # 添加高度数字
    plt.clabel(C, inline=True, fontsize=10)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# Image 图片
def draw_image():
    a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
                  0.365348418405, 0.439599930621, 0.525083754405,
                  0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)
    plt.imshow(a, cmap='bone', origin='lower', interpolation='nearest')
    plt.colorbar(shrink=0.92)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# 3D 数据
def draw_3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    # X, Y value
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
    R = np.sqrt(X ** 2 + Y ** 2)
    # height value
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim3d(-2, 2)
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
    plt.show()


# subplot 多图合一
def test_subplot():
    plt.figure()
    plt.subplot(2, 3, (1, 2))
    plt.plot([0, 1], [0, 1])
    plt.subplot(2, 3, 4)
    plt.plot([0, 1], [0, 1], color='red')
    plt.subplot(2, 3, 5)
    plt.plot([0, 1], [0, 1], color='blue')
    plt.subplot(1, 3, 3)
    plt.plot([0, 1], [0, 1], color='green')
    plt.show()


# subplot2grid 分格显示
def test_subplot2grid():
    plt.figure()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax1.plot([1, 2], [1, 2])
    ax1.set_title('ax1_title')
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax2.plot([1, 2], [1, 2], color='red')
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax3.plot([1, 2], [1, 2], color='green')
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax4.scatter([1, 2], [2, 2], color='blue')
    ax4.set_xlabel('ax4_x')
    ax4.set_ylabel('ax4_y')
    ax5 = plt.subplot2grid((3, 3), (2, 1))
    ax5.plot([1, 2], [1, 2], color='yellow')
    plt.show()


# gridspec 分格显示
def test_gridspec():
    import matplotlib.gridspec as gridspec
    plt.figure()
    gs = gridspec.GridSpec(3, 3)
    ax6 = plt.subplot(gs[0, :])
    ax6.plot([1, 2], [1, 2])
    ax7 = plt.subplot(gs[1, :2])
    ax7.plot([1, 2], [1, 2], color='red')
    ax8 = plt.subplot(gs[1:, 2])
    ax8.plot([1, 2], [1, 2], color='green')
    ax9 = plt.subplot(gs[-1, 0])
    ax9.plot([1, 2], [2, 2], color='blue')
    ax10 = plt.subplot(gs[-1, -2])
    ax10.plot([1, 2], [1, 2], color='yellow')
    plt.show()


# subplots 分格显示
def test_subplots():
    plt.figure()
    f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex='all', sharey='all')
    ax11.plot([1, 2], [1, 2])
    ax12.plot([1, 2], [1, 2], color='red')
    ax13.plot([1, 2], [1, 2], color='green')
    ax14.plot([1, 2], [1, 2], color='blue')
    plt.tight_layout()
    plt.show()


# plot in plot 图中图
def test_plot_in_plot():
    fig = plt.figure()

    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 3, 4, 2, 5, 8, 6]

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(x, y, color='red')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('title')

    left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(y, x, color='blue')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('title inside b')

    plt.axes([0.6, 0.2, 0.25, 0.25])
    plt.plot(y[::-1], x, color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('title inside g')

    plt.show()


# twinx 次坐标轴
def test_twinx():
    x = np.arange(0, 10, 0.1)
    y1 = 0.05 * x ** 2
    y2 = -1 * y1
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1, 'g-')  # green, solid line
    ax1.set_xlabel('X data')
    ax1.set_ylabel('Y1 data', color='g')
    ax2.plot(x, y2, 'b--')  # blue
    ax2.set_ylabel('Y2 data', color='b')
    plt.show()


# Animation 动画
def test_animation():
    from matplotlib import animation
    fig, ax = plt.subplots()
    x = np.arange(0, 2 * np.pi, 0.01)
    line, = ax.plot(x, np.sin(x))

    def animate(i):
        line.set_ydata(np.sin(x + i / 10.0))
        return line,

    def init():
        line.set_ydata(np.sin(x))
        return line,

    ani = animation.FuncAnimation(fig=fig,
                                  func=animate,
                                  frames=100,
                                  init_func=init,
                                  interval=20,
                                  blit=False)
    ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


if __name__ == '__main__':
    draw_scatter()
    # draw_bar()
    # draw_contours()
    # draw_image()
    # draw_3d()

    # test_subplot()
    # test_subplot2grid()
    # test_gridspec()
    # test_subplots()
    # test_plot_in_plot()
    # test_twinx()
    # test_animation()
