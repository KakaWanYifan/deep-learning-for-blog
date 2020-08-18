import numpy as np


# 损失函数
def loss(b, w, xArr, yArr):
    '''
    损失函数
    :param b: 偏置
    :param w: 权重
    :param xArr: xArr
    :param yArr: yArr
    :return: 损失函数的值
    '''
    # 损失值
    total_loss = 0
    for i in range(0, len(xArr)):
        x = xArr[i]
        y = yArr[i]
        total_loss = total_loss + (y - (w * x + b)) ** 2

    return total_loss / (float(len(xArr)))


# 一步，梯度
def step_gradient(b_cur, w_cur, xArr, yArr, lr):
    '''
    一步，梯度
    :param b_cur: 当前的偏置
    :param w_cur: 当前的权重
    :param xArr: xArr
    :param yArr: yArr
    :param lr: 学习率
    :return:
    '''
    b_gradient = 0
    w_gradient = 0
    n = float(len(xArr))
    for i in range(0, len(xArr)):
        x = xArr[i]
        y = yArr[i]
        w_gradient = w_gradient + (2 / n) * ((w_cur * x + b_cur) - y) * x
        b_gradient = b_gradient + (2 / n) * ((w_cur * x + b_cur) - y)
    b_new = b_cur - (lr * b_gradient)
    w_new = w_cur - (lr * w_gradient)

    return b_new, w_new


# 迭代更新
def gradient_descent(b_start, w_start, xArr, yArr, lr, iterations):
    '''
    迭代更新
    :param points: 数据点
    :param b_start: 偏置的初始值
    :param w_start: 权重的初始值
    :param lr: 学习率
    :param iterations: 最大迭代次数
    :return:
    '''
    b = b_start
    w = w_start
    for i in range(iterations):
        b, w = step_gradient(b_cur=b, w_cur=w, xArr=xArr, yArr=yArr, lr=lr)
    return b, w


# 等差数列
xArr = np.linspace(-10, 10, 100)
# 加上噪音
yArr = 2.0 * xArr + 1.0 + np.random.randn(len(xArr)) * 0.2

# 给b和w赋初值
b = 0
w = 0

# 初值的损失
print('初值的损失：', loss(b=b, w=w, xArr=xArr, yArr=yArr))

# 迭代更新 学习率0.001,迭代10000次
b, w = gradient_descent(b_start=b, w_start=w, xArr=xArr, yArr=yArr, lr=0.001, iterations=100000)
print('b：', b, '   w：', w)
print('训练后的损失：', loss(b=b, w=w, xArr=xArr, yArr=yArr))
