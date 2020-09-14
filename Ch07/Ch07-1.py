import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential, losses, optimizers

# 网络容器
network = Sequential([
    # 第一个卷积层, 6 个 3x3 卷积核
    layers.Conv2D(6, kernel_size=3, strides=1),
    # 高宽各减半的池化层
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 激活函数
    layers.ReLU(),
    # 第二个卷积层, 16 个 3x3 卷积核
    layers.Conv2D(16, kernel_size=3, strides=1),
    # 高宽各减半的池化层
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 激活函数
    layers.ReLU(),

    # 打平层，方便全连接层处理
    layers.Flatten(),

    # 全连接层，120 个节点
    layers.Dense(120, activation='relu'),
    # 全连接层，84 节点
    layers.Dense(84, activation='relu'),
    # 全连接层，10 个节点
    layers.Dense(10)])

network.build(input_shape=(4, 28, 28, 1))
# 统计网络信息
network.summary()

# 创建损失函数的类，在实际计算时直接调用类实例即可
criteon = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=0.001)


# 迭代更新
def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 插入通道维度，=>[b,28,28,1]
            x = tf.expand_dims(x, axis=3)
            # 前向计算，获得 10 类别的概率分布，[b, 784] => [b, 10]
            out = network(x)
            # 计算交叉熵损失函数，标量
            loss = criteon(y, out)

        print(epoch, step, loss.numpy())

        # 自动计算梯度
        grads = tape.gradient(loss, network.trainable_variables)
        # 自动更新参数
        optimizer.apply_gradients(zip(grads, network.trainable_variables))


def test():
    # 记录预测正确的数量，总样本数量
    correct, total = 0, 0
    for x, y in test_dataset:  # 遍历所有训练集样本
        # 插入通道维度，=>[b,28,28,1]
        x = tf.expand_dims(x, axis=3)
        # 前向计算，获得 10 类别的预测分布，[b, 784] => [b, 10]
        out = network(x)
        # 真实的流程时先经过 softmax，再 argmax
        # 但是由于 softmax 不改变元素的大小相对关系，故省去
        pred = tf.argmax(out, axis=-1)
        # 统计预测正确数量
        print(type(pred))
        print(type(y))
        correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
        # 统计预测样本总数
        total += x.shape[0]
        # 计算准确率
        print('test acc:', correct / total)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.0
    print(x_train.shape)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    y_train = tf.one_hot(y_train, depth=10)

    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

    train_epoch(20)
    test()
