import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, layers, datasets

class VAE(Model):

    def __init__(self):
        super(VAE, self).__init__()

        # 编码器
        self.fc1 = layers.Dense(128)
        # 均值
        self.fc2 = layers.Dense(10)
        # 方差
        self.fc3 = layers.Dense(10)

        # 解码器
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        # 均值
        mu = self.fc2(h)
        # 方差的log
        log_var = self.fc3(h)

        return mu, log_var


    def decoder(self, z):

        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var) ** 0.5
        z = mu + std * eps

        return z

    def call(self, inputs, training=None):
        # [b, 784] => [b, 10], [b, 10]
        mu, log_var = self.encoder(inputs)
        # Reparameterization Trick
        z = self.reparameterize(mu, log_var)

        x_hat = self.decoder(z)

        return x_hat, mu, log_var


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

    model = VAE()
    model.build(input_shape=(4, 784))
    model.summary()

    optimizer = tf.optimizers.Adam(lr=0.01)

    for epoch in range(5):
        for step, x in enumerate(train_db):
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, [-1, 784])

            with tf.GradientTape() as tape:
                x_rec_logits, mu, log_var = model(x)

                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
                rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

                kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_sum(kl_div) / x.shape[0]

                loss = rec_loss + 1. * kl_div

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))

    z = tf.random.normal((128, 10))
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() *255.
    x_hat = x_hat.astype(np.uint8)
    printImage(x_hat)
