import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential, layers, optimizers, losses

network = Sequential([
    # 卷积层1
    layers.Conv2D(filters=6, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1), padding="same"),
    layers.MaxPool2D(pool_size=(2, 2), strides=2),

    # 卷积层2
    layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same"),
    layers.MaxPool2D(pool_size=2, strides=2),

    # 卷积层3
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same"),

    layers.Flatten(),

    # 全连接层1
    layers.Dense(200, activation="relu"),

    # 全连接层2
    layers.Dense(10, activation="softmax")
])

if __name__ == '__main__':

    network.summary()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # 扩展为单通道
    x_train = tf.expand_dims(x_train, axis=3)
    x_test = tf.expand_dims(x_test, axis=3)
    print(x_train.shape)
    print(x_test.shape)

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(200)
    adam = optimizers.Adam()
    sparse_categorical_crossentropy = losses.SparseCategoricalCrossentropy()

    network.compile(optimizer=adam, loss=sparse_categorical_crossentropy, metrics=['accuracy'])
    network.fit(db_train, epochs=10)

    # 测试模型
    loss, accuracy = network.evaluate(db_test)
    print(loss)
    print(accuracy)

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_test[i], cmap='gray')
    plt.show()

    # 预测前25张图片结果
    result = network.predict(x_test)[0:25]
    pred = tf.argmax(result, axis=1)
    pred_list = []
    for item in pred:
        pred_list.append(item.numpy())
    print(pred_list)

