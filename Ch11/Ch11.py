import tensorflow as tf
import numpy as np
import glob
import Ch11Util
import matplotlib.pyplot as plt
from tensorflow.keras import Model, layers, losses

tf.random.set_seed(1)
np.random.seed(1)


# 生成器
class Generator(Model):

    def __init__(self):
        super(Generator, self).__init__()

        # z: [b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => [b, 64, 64, 3]
        self.fc = layers.Dense(3 * 3 * 512)

        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):
        # [z, 100] => [z, 3*3*512]
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        x = tf.tanh(x)

        return x


# 鉴别器
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # [b, 64, 64, 3] -> [b, 1]

        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')

        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        # [b, h, w ,c] => [b, -1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        # [b, h, w, c] => [b, -1]
        x = self.flatten(x)

        # [b, -1] => [b, 1]
        logits = self.fc(x)

        return logits


def celoss_ones(logits):
    y = tf.ones_like(logits)
    loss = losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    y = tf.zeros_like(logits)
    loss = losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


# 生成器的损失函数
def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 在训练生成网络时,需要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_image, is_training)
    # 计算生成图片与1 之间的误差
    loss = celoss_ones(d_fake_logits)

    return loss


# 鉴别器的损失函数
def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 判定生成图片
    d_fake_logits = discriminator(fake_image, is_training)

    # 判定真实图片
    d_real_logits = discriminator(batch_x, is_training)
    # 真实图片与 1 之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与 0 之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)

    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss


def paintImg(images):
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        x = (images[i] + 1.0) * 127.5
        x = tf.cast(x=x, dtype=tf.int32)
        plt.imshow(x)
    plt.show()


if __name__ == '__main__':

    g = Generator()
    z = tf.random.normal([2, 100])
    print(g(z).shape)

    img_paths = glob.glob('./Ch11-Data/*.jpg')
    print(len(img_paths))

    dataset, img_shape, len_dataset = Ch12Util.make_anime_dataset(img_paths=img_paths, batch_size=128, resize=64)
    print(dataset, img_shape, len_dataset)
    sample = next(iter(dataset))
    print(sample.shape)

    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, 100))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimizer = tf.optimizers.Adam(learning_rate=0.002, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=0.002, beta_1=0.5)

    for epoch in range(1000000):

        batch_z = tf.random.uniform([128, 100], minval=-1.0, maxval=1.0)
        batch_x = next(db_iter)

        # 训练 D
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, True)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # 训练 G
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, True)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))

            z = tf.random.uniform([25, 100])
            fake_image = generator(z, training=False)
            paintImg(fake_image)
