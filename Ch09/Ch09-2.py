import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(1)
np.random.seed(1)


class RNN(keras.Model):

    def __init__(self, units):
        super(RNN, self).__init__()

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        self.rnnlayer = keras.Sequential([
            layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.SimpleRNN(units, dropout=0.5, unroll=True)
        ])

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 100]
        x = inputs
        # embedding: [b, 100] => [b, 100, 150]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b, 100, 150] => [b, 64]
        x = self.rnnlayer(x, training=training)

        # out: [b, 64] => [b, 1]
        x = self.outlayer(x)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


if __name__ == '__main__':
    batch_size = 128

    # the most frequest words
    total_words = 10000
    max_review_len = 80
    embedding_len = 100
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
    # x_train:[b, 80]
    # x_test: [b, 80]
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.batch(batch_size, drop_remainder=True)
    print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
    print('x_test shape:', x_test.shape)

    units = 64
    epochs = 4

    model = RNN(units)
    # model.build(input_shape=(4,100))
    # model.summary()
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)
