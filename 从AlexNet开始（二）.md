# 从AlexNet开始（二）

本篇博客将会介绍如何改进AlexNet的网络结构使其应用于CIFAR-10数据集的图像分类，并给出大部分代码，代码使用python语言与tensorflow框架。

------

## CIFAR-10数据集简介

CIFAR数据集是由深度学习三大牛之一的Hinton大佬的两个大弟子Alex Krizhevsky与Ilya Sutskever收集的一个用于普适物体识别的数据集，其中CIFAR是加拿大政府牵头投资的一个先进科学项目研究所。CIFAR数据集共分为两部分，一部分是CIFAR-10，另一部分是CIFAR-100 。

CIFAR-10由60000张32*32分辨率的RGB彩色照片组成，顾名思义，这60000张照片共有10个类别，其中训练集包括50000张图片，测试集包括10000张图片，训练集的图片按类别分好，而测试集的图片顺序是打乱的。

![cifar10](e:\blog\2\cifar.png)

CIFAR-100数据集则共有100个分类，其中100个分类包含在10个大类之中，每个大类有10个小类。CIFAR-100中包含的图片同样也是50000张用来训练，10000张用来测试。

具体的CIFAR数据集下载与解压，可以参考Alex做的CIFAR数据集的网站，链接如下：

> http://www.cs.toronto.edu/~kriz/cifar.html

------

## Tensorflow简介

Tensorflow是由谷歌开发的，用于机器学习与深度神经网络计算的开源框架。

在Tensorflow中，数据不再是传统的多维数组与变量，程序也不再是由传统的结构组成。Tensorflow采用数据流图的方式来描述程序中的运算，在数据流图中，节点（node）表示对变量的操作与数学计算或是数据的起点终点，线（edge）表示节点之间的输入输出关系，而在线中流动的，就是输送数据的多维数组，也即张量（tensor）。

![tensorflow](E:\zym\ML_ppt\tensorflow.gif)

一旦构建好数据流图，tensors将会在节点之间流动，并异步并行地执行数据流图所定义的运算

------

## Tensorflow代码解释

### 代码概述

Tensorflow的代码共分为4个文件，分别为

| Code             | Description                              |
| :--------------- | ---------------------------------------- |
| cifar10.py       | 主程序，决定模型训练还是验证                           |
| cifar10_input.py | 输入程序，包括数据增强、建立队列、打乱顺序等                   |
| cifar10_model.py | 模型搭建，包括inference, train, loss, validation等函数 |
| cifar10_eval.py  | 验证程序，验证模型在测试集上的精确度                       |

重点说下cifar10_model.py的内容，输入部分的代码可以参考我之前的博客，如何使用队列输入数据。全部代码会上传到我的github上，地址为

> https://github.com/zym1119

### cifar10_model.py

本代码中共包含如下函数

| Functions                  | Description       |
| -------------------------- | ----------------- |
| inference                  | 模型推测              |
| loss                       | 计算损失函数            |
| train                      | 模型训练              |
| validation                 | 模型验证              |
| variable_on_cpu            | 创建存储在CPU内存上的变量    |
| variable_with_weight_decay | 创建变量并计算L2_loss    |
| add_loss_summaries         | summary所有的loss并求和 |

inference函数定义了前向模型，描述了卷积神经网络的结构。在函数中，输入一个batch的图片，以及dropout的keep_prob，最终得到计算出的logits。

```python
def inference(images, keep_prob):
	images = tf.cast(images, tf.float32)

    # conv 1
    # 5*5 convolution with stride 2, channel 32
    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights', [5, 5, 3, 32], stddev=0.1, wd=0.0)
        bias = variable_on_cpu('bias', [32], tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], 'SAME')
        pre_acti1 = tf.add(conv, bias)
        conv1 = tf.nn.relu(pre_acti1, name=scope.name)

    # local response normalization 1
    norm1 = tf.nn.lrn(conv1, 5, 2, 0.0001, 0.75, name='norm1')

    # max pooling 1
    pool1 = tf.nn.max_pool(norm1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='max_pool1')

    # conv 2
    # 3*3 convolution, channel 48
    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights', [3, 3, 32, 48], 0.1, 0.0)
        bias = variable_on_cpu('bias', [48], tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], 'SAME')
        pre_acti2 = tf.add(conv, bias)
        conv2 = tf.nn.relu(pre_acti2, name=scope.name)

    # conv 3
    # 3*3 convolution, channel 64
    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_weight_decay('weights', [3, 3, 48, 64], 0.1, 0.0)
        bias = variable_on_cpu('bias', [64], tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], 'SAME')
        pre_acti3 = tf.add(conv, bias)
        conv3 = tf.nn.relu(pre_acti3, name=scope.name)

    # conv 4
    # 3*3 convolution, channel 64
    with tf.variable_scope('conv4') as scope:
        kernel = variable_with_weight_decay('weights', [3, 3, 64, 64], 0.1, 0.0)
        bias = variable_on_cpu('bias', [64], tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], 'SAME')
        pre_acti4 = tf.add(conv, bias)
        conv4 = tf.nn.relu(pre_acti4, name=scope.name)

    # local response normalization 2
    norm2 = tf.nn.lrn(conv4, 5, 2, 0.0001, 0.7, name='norm2')

    # max pooling 2, size (128, 3, 3, 64)
    pool2 = tf.nn.max_pool(norm2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='max_pool2')

    # fully connected 1
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool2, [-1, 3*3*64])
        weights = variable_with_weight_decay('weights', [3*3*64, 256], 0.01, 0.0005)
        bias = variable_on_cpu('bias', [256], tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights)+bias, name=scope.name)

    # drop out 1
    fc1_dropout1 = tf.nn.dropout(fc1, keep_prob[0], name='dropout1')

    # fully connected 2
    with tf.variable_scope('fc2') as scope:
        weights = variable_with_weight_decay('weights', [256, 256], 0.01, 0.0005)
        bias = variable_on_cpu('bias', [256], tf.constant_initializer(0.0))
        fc2 = tf.nn.relu(tf.matmul(fc1_dropout1, weights) + bias, name=scope.name)

    # drop out 2
    fc2_dropout = tf.nn.dropout(fc2, keep_prob[1], name='dropout2')

    # softmax linear
    with tf.variable_scope('softmax_linear') as scope:
        weights = variable_with_weight_decay('weights', [256, 10], 0.1, 0.0005)
        bias = variable_on_cpu('bias', [10], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc2_dropout, weights), bias, name=scope.name)

    return softmax_linear
```

考虑到CIFAR数据集图片的大小与图片预处理步骤，相对于原始的AlexNet，做出了以下的改动：

1. 将5个卷积层减少为4个，并相应地减少每个卷积层的通道数，但通道之间的比例尽量保持原始AlexNet各卷积层通道数的比例；
2. 权重的初始化中，将kernel的均值改为0.1（AlexNet为0.01），因为0.01对于白化后的像素值过小，会导致模型无法学习或学习及其缓慢
3. 偏置的初始化中，将特定层的偏置初始化为0.1（AlexNet为1），同样是因为对图像做了白化处理
4. 全连接层的大小调整为256而非4096



loss函数计算了输出的交叉熵损失与权重的l2正则化损失并相加

```python
def loss(logits, labels):
    """
    Calculates loss with L2 regularization
    :param logits: output of softmax linear
    :param labels: a batch of labels, 1D tensor of size [batch_size]
    :return:
        total_loss: loss tensor
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels,                                                                  name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss
```

权重的l2损失是在variable_with_weight_decay中定义的，但加入了名为"losses"的collection，因此只需要使用tf.get_collection()函数得到该collection并相加即可



train函数定义了训练一步模型的Operation，完成了对模型中数据的汇总、梯度的计算与loss的优化

```python
def train(total_loss, global_steps):

    # define learning rate
    num_batch_per_epoch = 50000/128     # 390
    lr_decay_steps = int(num_batch_per_epoch*10)
    lr = tf.train.exponential_decay(0.1, global_steps, lr_decay_steps, 0.1, staircase=True)
    # summary learning rate
    loss_averages_op = _add_loss_summaries(total_loss)

    # add histograms for trainable variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # compute gradients
    with tf.control_dependencies(control_inputs=[loss_averages_op]):
        apply_gradient_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(total_loss, global_step=global_steps)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999, global_steps)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op, lr
```

第6行把learning_rate定义为一个指数衰减的tensor，lr_decay_steps表示每次学习速率减小所需的迭代次数。

第15行的tf.control_dependencies()函数表示tensorflow中的控制依赖关系，在执行完括号中的操作后才会执行with内部的操作。



validation函数定义了函数的验证方式，即把logits与labels进行对比，得到模型的正确率，当训练的时候运行该函数，则得到当前模型在训练集上的正确率，当把模型训练好后运行cifar10_eval.py，则得到当前模型在测试集上的正确率。

```python
def validation(logits, labels):
    correct_predict = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)
    accuracy = tf.reduce_mean(correct_predict)
    tf.summary.scalar('accuracy', accuracy)
    return tf.cast(accuracy, tf.float32)
```

tf.nn.in_top_k()函数得到logtis对应位置上labels中的label是否是前k大的数，是则返回True否则返回False。



variable_with_weight_decay()函数创建一个具有权重衰减的变量

```python
def variable_with_weight_decay(name, shape, stddev, wd):
    """
    Creates an initialized variable with weight decay
    :param name: name of the variable
    :param shape: shape of the variable, list of ints
    :param stddev: standard deviation of a truncated Gaussian
    :param wd: weight decay factor, multiplied with the L2 loss of the weight
    :return:
        var: variable tensor
    """
    dtype = tf.float32
    var = variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
```

这个版本的代码我使用了xavier_initializer，AlexNet中应该使用高斯分布的initializer。

------

## 结束语

这部分代码算是我接触tensorflow以后写的第一个较大型的神经网络，折腾了好几天，部分代码参考了tensorflow的官方入门手册中的代码，在看了官方的代码后，确实让我的代码风格更加规范，推荐新手都看下Google的程序员是怎么写的。

刚开始照搬了AlexNet的结构，模型完全不收敛，后来自己开始琢磨，发现在run sess的时候没有run训练的op，于是加入了_ = sess.run(train_op, feed_dict={keep_prob: [0.5, 0.5]})以后，才看到global_step的更新。但又发现loss在迭代几万次后几乎没变，accuracy也在0.1左右浮动，改了好几天都不知道为什么，后来在教研室同学的帮助下调了调参，才发现由于图像预处理时加入的白化改变了像素的大小，完全照搬AlexNet的参数初始化是不行的。最后改了一改终于跑了出来，最好的测试集准确率刚过80，但现在代码又被我改的不像样子了，当时参数是什么网络结构是什么也不记得，但对于新手来说拿CIFAR-10来练练手熟悉tensorflow还是足够了。