import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers, Sequential, datasets, optimizers


# BasicBlock
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, strides=1):
        """

        :param filter_num: 卷积核的数量
        :param strides: 步长，默认为1
        """
        # 初始化函数，调用父类的初始化方法
        super(BasicBlock, self).__init__()

        # f(x)包含了2个普通卷积层
        # 创建卷积层 1
        self.conv1 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # 创建卷积层 2
        self.conv2 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if strides != 1:
            # 插入 identity 层
            self.identity = Sequential()
            self.identity.add(layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=strides))
        else:
            # 否则,直接连接
            self.identity = lambda x: x

    def call(self, inputs, training=None):
        """

        :param inputs:
        :param training:
        :return:
        """
        # 前向传播函数
        # 通过第一个卷积层
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        # 通过第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        # 输入通过 identity()转换
        idVal = self.identity(inputs)

        # f(x) + x 运算
        output = layers.add([out, idVal])

        # 再通过激活函数并返回
        # 特别注意：relu用了两次
        output = tf.nn.relu(output)
        return output

class ResNet(keras.Model):

    def __init__(self,layer_dims,num_class=100):
        """

        :param layer_dims:
        """
        super(ResNet,self).__init__()
        self.stem = Sequential([layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')])

        self.layer1 = self.build_resblock(filter_num=64,blocks=layer_dims[0])
        self.layer2 = self.build_resblock(filter_num=128,blocks=layer_dims[0],strides=2)
        self.layer3 = self.build_resblock(filter_num=256,blocks=layer_dims[0],strides=2)
        self.layer4 = self.build_resblock(filter_num=512,blocks=layer_dims[0],strides=2)

        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(units=num_class)


    def call(self, inputs, training=None):
        """

        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = self.stem(inputs,training=training)

        x = self.layer1(x,training=training)
        x = self.layer2(x,training=training)
        x = self.layer3(x,training=training)
        x = self.layer4(x,training=training)

        x = self.avgpool(x)

        x = self.fc(x)

        return x


    def build_resblock(self,filter_num,blocks,strides=1):
        """

        :param filter_num:
        :param blocks:
        :param strides:
        :return:
        """
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num=filter_num,strides=strides))

        for _ in range(blocks):
            res_blocks.add(BasicBlock(filter_num=filter_num,strides=1))

        return res_blocks

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

    # 减少数据量，这要耗费时间少。毕竟只是demo
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_test = x_test[:2000]
    y_test = y_test[:2000]

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

    model = ResNet(layer_dims=[2,2,2,2])
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    epochs = 1
    for epoch in range(epochs):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                # [b,32,32,3] => [b,100]
                logits = model(x)

                # one hot
                y_onehot = tf.one_hot(y, depth=100)

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:
                print(epoch, step, 'loss: ', loss.numpy())

    total_num = 0
    total_correct = 0
    for x, y in db_test:
        logits = model(x)
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
