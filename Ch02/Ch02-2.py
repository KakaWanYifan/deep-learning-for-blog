import tensorflow as tf
from tensorflow.keras import datasets

# 准备数据
(xs, ys), _ = datasets.mnist.load_data()

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.0
ys = tf.convert_to_tensor(ys, dtype=tf.int32)
ys = tf.one_hot(ys, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(200)

# 定义模型

w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
b1 = tf.Variable(tf.zeros([512]))
w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1))
b2 = tf.Variable(tf.zeros([256]))
w3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 0.001


# 迭代更新
def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)

            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2 @ w3 + b3
            out = tf.nn.relu(out)

            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)

        print(epoch, step, 'loss:', float(loss))

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print(grads)
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])


for epoch in range(10):
    train_epoch(epoch)
