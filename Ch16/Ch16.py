import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential, layers, optimizers, losses

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.expand_dims(x_train, axis=3)
x_test = tf.expand_dims(x_test, axis=3)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(200)

# 主要讨论这行之后的代码
# 定义一个网络容器
network = Sequential([
    # 网络层
    layers.Conv2D(filters=6, kernel_size=(5, 5), activation="relu", padding="same"),
    layers.MaxPool2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same"),
    layers.MaxPool2D(pool_size=2, strides=2),
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(200, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 构建网络
network.build(input_shape=(None, 28, 28, 1))
# 打印网络信息
network.summary()
# 优化器
adam = optimizers.Adam()
# 损失
sparse_categorical_crossentropy = losses.SparseCategoricalCrossentropy()
# 装配
network.compile(optimizer=adam, loss=sparse_categorical_crossentropy, metrics=['accuracy'])
# 训练
network.fit(db_train, epochs=5)

print('\n' * 3)

# 评估
loss, accuracy = network.evaluate(db_test)
print(loss, accuracy)
# 预测
result = network.predict(x_test)[0:25]
pred = tf.argmax(result, axis=1)
print(pred)

print('\n' * 3)

# 第一种模型保存和加载方法
network.save_weights('weights.ckpt')
print('saved weights.')
# 删除网络对象
del network
# 重新创建相同的网络结构
network = Sequential([
    # 网络层
    layers.Conv2D(filters=6, kernel_size=(5, 5), activation="relu", padding="same"),
    layers.MaxPool2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same"),
    layers.MaxPool2D(pool_size=2, strides=2),
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(200, activation="relu"),
    layers.Dense(10, activation="softmax")
])
network.build(input_shape=(None, 28, 28, 1))
network.compile(optimizer=adam, loss=sparse_categorical_crossentropy, metrics=['accuracy'])
network.load_weights('weights.ckpt')
print('loaded weights!')
# 看看第一种方法的对不对
print(network.evaluate(db_test))

print('\n' * 3)

# 第二种模型保存和加载方法
network.save('model.h5')
print('saved total model.')
del network
network = keras.models.load_model('model.h5')
# 看看第二种对不对
print(network.evaluate(db_test))

print('\n' * 3)

# 第三种模型保存和加载方法
tf.saved_model.save(network, 'model-savedmodel')
print('saving savedmodel.')
del network
print('load savedmodel from file.')
network = tf.saved_model.load('model-savedmodel')
f = network.signatures['serving_default']
print(f(conv2d_3_input=tf.ones([1, 28, 28, 1])))
