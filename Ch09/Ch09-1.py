import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(1)
np.random.seed(1)

class RNN(keras.Model):

    def __init__(self, units):
        super(RNN, self).__init__()

        # [b, 64]
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]

        # transform text to embedding representation
        # [b, 100] => [b, 100, 150]
        self.embedding = layers.Embedding(input_dim=total_words, output_dim=embedding_len, input_length=max_review_len)

        # SimpleRNNCell
        # units=64
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)

        # 全连接层
        # [b, 100, 150] => [b, 64] => [b, 1]
        self.out = layers.Dense(1)

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
        # [b, 100, 150] => [b, 64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            # word: [b, 150]
            # h1 = x*wxh+h0*whh
            # out0: [b, 64]
            out0, state0 = self.rnn_cell0(word, state0, training)
            # out1: [b, 64]
            out1, state1 = self.rnn_cell1(out0, state1, training)

        # out: [b, 64] => [b, 1]
        x = self.out(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


if __name__ == '__main__':
    # 批处理，批训练，发挥现代CPU和GPU的优势
    batch_size = 128

    # 词汇表大小
    total_words = 10000

    # 每个句子的最大长度
    max_review_len = 100

    # 每个词的表示向量的维度数
    embedding_len = 150

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.batch(batch_size, drop_remainder=True)

    units = 64
    epochs = 4

    model = RNN(units)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'],
                  experimental_run_tf_function=False)
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    print('evaluate:')
    model.evaluate(db_test)
