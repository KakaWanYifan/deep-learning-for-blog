# %%

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# 全连接网络层
class Layer:

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """

        :param n_input: 输入节点数
        :param n_neurons: 输出节点数
        :param activation: 激活函数类型
        :param weights: 权重
        :param bias: 偏置
        """
        # 输入节点数
        self.n_input = n_input
        # 输出节点数
        self.n_neurons = n_neurons
        # 激活函数类型,如’sigmoid’
        self.activation = activation
        # 通过正态分布初始化网络权值，初始化非常重要，不合适的初始化将导致网络不收敛
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        # 初始化偏置
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1

        # 激活函数的输出值
        self.outputs = None
        # 用于计算当前层 delta 的中间变量
        self.error = None
        # 记录当前层的 delta 变量，给前一层计算梯度用
        self.delta = None

    def calculate_outputs(self, inputs):
        """

        :param inputs: 输入值
        :return: 输出值，last_activation
        """
        # x@w + b
        z = np.dot(inputs, self.weights) + self.bias
        # z 作为激活函数的自变量
        # 无激活函数,直接返回
        if self.activation is None:
            self.outputs = z
        # sigmoid 激活函数
        elif self.activation.lower() == 'sigmoid':
            self.outputs = 1 / (1 + np.exp(-z))
        # tanh 激活函数
        elif self.activation.lower() == 'tanh':
            self.outputs = np.tanh(z)
        # ReLU 激活函数
        elif self.activation.lower() == 'relu':
            self.outputs = np.maximum(z, 0)
        # LeakyReLU 激活函数
        # p 值 采用TensorFlow的默认P值，0.2
        elif self.activation.lower() == 'leakyrelU':
            if z >= 0:
                self.outputs = z
            else:
                self.outputs = z * 0.2

        return self.outputs

    # 计算激活函数的导数
    def activation_derivative(self, z):
        """

        :param z: 激活函数的值
        :return: 导函数的值
        """
        # 无激活函数,导数为 1
        if self.activation is None:
            return np.ones_like(z)
        # Sigmoid 的导数
        elif self.activation.lower() == 'sigmoid':
            return z * (1 - z)
        # tanh 的导数
        elif self.activation.lower() == 'tanh':
            return 1 - z ** 2
        # ReLU 的导数
        elif self.activation.lower() == 'relu':
            grad = np.array(z, copy=True)
            grad[z > 0] = 1.
            grad[z <= 0] = 0.
            return grad
        # LeakyReLU 的导数
        # p 值 采用TensorFlow的默认P值，0.2
        elif self.activation.lower() == 'leakyrelU':
            grad = np.array(z, copy=True)
            grad[z > 0] = 1.
            grad[z <= 0] = 0.2
            return grad

        return z


# 神经网络模型
class NeuralNetwork:
    def __init__(self):
        # 网络层列表
        self.layers = []

    # 新增网络层
    def add_layer(self, layer):
        """

        :param layer: 网络层
        :return: 新的网络层列表
        """
        self.layers.append(layer)

    # 删除网络层
    def remove_layer(self, layer):
        """

        :param layer: 网络层
        :return: 新的网络层列表
        """
        self.layers.remove(layer)

    # 前向传播
    def feed_forward(self, inputs):
        """

        :param inputs: 整个神经网络的输入
        :return: 整个神经网络的输出
        """
        # 大写的X代表矩阵
        X = inputs
        for layer in self.layers:
            X = layer.calculate_outputs(X)

        return X

    # 反向传播
    def backpropagation(self, inputs, trues, lr):
        """

        :param inputs: 整个神经网络的输入
        :param trues: 真实值
        :param lr: 学习率
        :return:
        """
        # 1. 计算整个神经网络的输出。
        outputs = self.feed_forward(inputs)
        # 反向循环
        # 分别计算输出层的delta 和 隐藏层的delta并存起来。
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # 计算输出层的delta
            # \delta_k^{K} = (o_k - t_k)(o_k - o_k^2)
            if layer == self.layers[-1]:
                layer.error = trues - outputs
                layer.delta = layer.error * layer.activation_derivative(outputs)
            # 隐藏层的delta
            # \delta_j^J = (o_j - o_j^2)\sum_{k = 1}^{K}\delta_k^{K} w_{jk}
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.activation_derivative(layer.outputs)

        # 更新权重
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # 寻找前一层的输出，即当前层的输入
            o_i = np.atleast_2d(inputs if i == 0 else self.layers[i - 1].outputs)
            # 将其与delta相乘。这就是梯度
            grads = layer.delta * o_i.T
            layer.weights = layer.weights + grads * lr

    # 网络训练函数
    def train(self, x_train, y_train, lr, max_epochs):
        """

        :param x_train: x 训练数据
        :param y_train: y 训练数据
        :param lr: 学习率
        :param max_epochs: 训练机的最大循环迭代次数
        :return:
        """
        # one-hot 编码
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        # 训练 1000 个 epoch
        for i in range(max_epochs):
            # 一次训练一个样本
            for j in range(len(x_train)):
                self.backpropagation(x_train[j], y_onehot[j], lr)
            if i % 10 == 0:
                # 打印出 MSE Loss
                mse = np.mean(np.square(y_onehot - self.feed_forward(x_train)))
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))

    def test(self, x_test, y_test):
        """

        :param x_test: x 测试数据
        :param y_test: y 测试数据
        :return:
        """
        # one-hot 编码
        y_onehot = np.zeros((y_test.shape[0], 2))
        y_onehot[np.arange(y_test.shape[0]), y_test] = 1
        # 整体训练完成之后
        mse = np.mean(np.square(y_onehot - self.feed_forward(x_test)))
        print('Finish!!!, MSE: %f' % float(mse))


def generate_data(n_samples, test_rate, noise):
    """

    :param n_samples: 样本总数
    :param test_rate: 测试数据所占比例
    :param noise: 噪音数据
    :return:
    """
    x, y = make_moons(n_samples=n_samples, noise=noise)
    return train_test_split(x, y, test_size=test_rate)


if __name__ == '__main__':
    x_data_train, x_data_test, y_data_train, y_data_test = generate_data(n_samples=2000, test_rate=0.25, noise=0.2)

    nn = NeuralNetwork()
    nn.add_layer(Layer(n_input=2, n_neurons=16, activation='sigmoid'))
    nn.add_layer(Layer(n_input=16, n_neurons=64, activation='sigmoid'))
    nn.add_layer(Layer(n_input=64, n_neurons=32, activation='sigmoid'))
    nn.add_layer(Layer(n_input=32, n_neurons=2, activation='sigmoid'))
    nn.train(x_train=x_data_train, y_train=y_data_train, lr=0.01, max_epochs=1000)
    nn.test(x_test=x_data_test, y_test=y_data_test)
