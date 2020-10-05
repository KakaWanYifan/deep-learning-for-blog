import tensorflow as tf
from tensorflow.keras import layers, Sequential, datasets, optimizers

conv_pool_layers = [

    # 第一个单元
    layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

    # 第二个单元
    layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

    # 第三个单元
    layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

    # 第四个单元
    layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

    # 第五个单元
    layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
]

conv_pool_net = Sequential(conv_pool_layers)
conv_pool_net.build(input_shape=[None, 32, 32, 3]);


fc_layers = [
    layers.Dense(units=512, activation=tf.nn.relu),
    layers.Dense(units=256, activation=tf.nn.relu),
    layers.Dense(units=100)
]
fc_net = Sequential(fc_layers)
fc_net.build(input_shape=[None, 512])


def preprocess(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.int32)

    return x, y


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    # 把一个维度挤压掉
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(buffer_size=1000).map(preprocess).batch(200)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).batch(200)

    sample = next(iter(db_train))
    print(sample[0].shape)
    print(sample[1].shape)

    optimizer = optimizers.Adam(learning_rate=0.0001)
    variables = conv_pool_net.trainable_variables + fc_net.trainable_variables

    epochs = 1
    for epoch in range(epochs):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                # [b,32,32,3] => [b,1,1,512]
                out = conv_pool_net(x)
                # [b,1,1,512] => [b,512]
                out = tf.reshape(out, [-1, 512])
                # [b,512] => [b,100]
                logits = fc_net(out)

                # one hot
                y_onehot = tf.one_hot(y, depth=100)

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 10 == 0:
                print(epoch, step, 'loss: ', loss.numpy())

    total_num = 0
    total_correct = 0
    for x, y in db_test:
        out = conv_pool_net(x)
        out = tf.reshape(out, [-1, 512])
        logits = fc_net(out)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_num = total_num + x.shape[0]
        total_correct = total_correct + correct.numpy()

    print(total_num)
    print(total_correct)
    acc = total_correct / total_num
    print(acc)
