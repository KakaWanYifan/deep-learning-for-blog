import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
from tensorflow.keras import layers, Sequential
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

x, y = make_moons(n_samples=2000, noise=0.2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 可视化的 x 坐标范围为[-2, 3]
xx = np.arange(-2, 3, 0.01)
# 可视化的 y 坐标范围为[-1.5, 2]
yy = np.arange(-1.5, 2, 0.01)
# 生成 x-y 平面采样网格点，方便可视化
XX, YY = np.meshgrid(xx, yy)


def mscatter(x, y, ax=None, m=None, **kw):
    """

    :param x:
    :param y:
    :param ax:
    :param m:
    :param kw:
    :return:
    """
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def make_plot(X, y, plot_name, XX=None, YY=None, preds=None):
    """

    :param X:
    :param y:
    :param plot_name:
    :param XX:
    :param YY:
    :param preds:
    :return:
    """
    axes = plt.gca()
    axes.set_xlim([-2, 3])
    axes.set_ylim([-1.5, 2])
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=20, fontproperties='SimHei')
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if XX is not None and YY is not None and preds is not None:
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.08, cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色m=markers
    markers = ['o' if i == 1 else 's' for i in y.ravel()]
    mscatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, cmap=plt.cm.Spectral, edgecolors='none', m=markers, ax=axes)
    plt.show()


def validation_split_demo():
    """

    :return:
    """
    for n in range(3):
        validation_split = n * 0.1

        model = Sequential()
        model.add(layers.Dense(16, input_dim=2, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, validation_split=validation_split, verbose=1)

        preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
        title = "validation_split：{0}".format(validation_split)
        make_plot(x_train, y_train, title, XX, YY, preds)

if __name__ == '__main__':
    validation_split_demo()
