import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Sequential, layers, datasets


class AE(Model):
    # 包含了Encoder和Decoder
    def __init__(self):
        super(AE, self).__init__()

        # 创建Encoders网络
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(20)
        ])

        # 创建Decoders网络
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        # 前向传播函数
        # 编码获得隐藏向量h,[b, 784] => [b, 20]
        h = self.encoder(inputs)
        # 解码获得重建图片，[b, 20] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat


def printImage(images):
    plt.figure(figsize=(10.0, 8.0))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

    plt.show()


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0
    # we do not need label
    train_db = tf.data.Dataset.from_tensor_slices(x_train)
    train_db = train_db.shuffle(128 * 5).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices(x_test)
    test_db = test_db.batch(128)

    model = AE()
    model.build(input_shape=(None, 784))
    model.summary()

    optimizer = tf.optimizers.Adam(lr=0.01)

    for epoch in range(5):
        for step, x in enumerate(train_db):
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, [-1, 784])

            with tf.GradientTape() as tape:
                x_rec_logits = model(x)

                rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
                rec_loss = tf.reduce_mean(rec_loss)

            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, float(rec_loss))

    # evaluation
    x = next(iter(test_db))
    logits = model(tf.reshape(x, [-1, 784]))
    x_hat = tf.sigmoid(logits)
    # [b, 784] => [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])

    # [b, 28, 28] => [2b, 28, 28]
    # 输入的前 50 张+重建的前 50 张图片合并
    x_concat = tf.concat([x[:10], x_hat[:10]], axis=0)
    # 恢复为 0-255 的范围
    x_concat = x_concat.numpy() * 255.
    # 转换为整型
    x_concat = x_concat.astype(np.uint8)
    printImage(x_concat)
