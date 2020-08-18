import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

# 准备数据
(xs, ys), _ = datasets.mnist.load_data()

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.0
ys = tf.convert_to_tensor(ys, dtype=tf.int32)
ys = tf.one_hot(ys, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(200)

# 定义模型

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='relu')
])

optimizer = optimizers.SGD(learning_rate=0.001)


# 迭代更新
def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [n,28,28] -> [n,784]
            x = tf.reshape(x, (-1, 28 * 28))
            # 计算输出
            out = model(x)
            # 计算损失
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # 梯度优化
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(epoch, step, loss.numpy())


for epoch in range(10):
    train_epoch(epoch)
