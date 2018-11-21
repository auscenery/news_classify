#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf


class TextCnnModel:
    """
        text-cnn 模型, author: Yoon Kim, 参考模型:
        "Convolutional Neural Networks for Sentence Classification",
        这里只采用one-channel, not multi-channels
    """
    def __init__(self, configs):
        """
        :param configs: bunch.Bunch类型,dict的子类别
        """
        # 超参数设置
        self.embedding_size = configs.embedding_size      # 词向量维度
        self.sequence_length = configs.sequence_length   # 序列长度
        self.labels_num = configs.labels_num             # 类别数
        self.num_filters = configs.num_filters           # 卷积核数目,即提取特征数目
        self.filter_sizes = configs.filter_sizes         # 卷积核大小
        self.vocabulary_size = configs.vocabulary_size   # 词汇表达小
        self.fc_hidden_size = configs.fc_hidden_size   # 全连接层神经元
        self.dropout_keep_prob = configs.dropout_keep_prob # dropout保留比例
        self.learning_rate = configs.learning_rate       # 学习率
        self.batch_size = configs.batch_size             # 每批训练大小
        self.num_epochs = configs.num_epochs             # 总迭代轮次

        # 普通参数设置
        self.print_per_batch = configs.print_per_batch   # 每多少轮打印一次训练结果信息
        self.summary_save_per_batch = configs.summary_save_per_batch  # 每多少轮存入tensorboard
        # 提前退出的阈值
        self.threshold_valid_evaluation_batch = configs.threshold_valid_evaluation_batch

        # 是否采用gpu, True, 采用gpu, 反之采用cpu
        self.gpu_flags = configs.gpu_flags

        # 用于自动计数步数
        with tf.variable_scope("global_step"):
            self.global_step_tensor = tf.Variable(
                0, trainable=False, name="global_step"
            )

        # 构造cnn模型
        self.bulid_cnn()

    def bulid_cnn(self):
        print("开始搭建cnn模型:")

        """define the cnn structure"""
        # convenient to modify, set to 0.5 when train, however,
        # set to 1 where valid and test
        self.keep_probability = tf.placeholder(
            dtype=tf.float32,
            name="keep_probability"
        )

        # firstly, define zero layers: input layer ,
        # input_x, [batch_size, seq_length]
        # input_y, [batch_size, labels_num]
        # None means batch_size can be arbitrary magnitudes
        self.input_x = tf.placeholder(
            dtype=tf.int32,
            shape=[None, self.sequence_length],
            name="input_x"
        )
        self.input_y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.labels_num],
            name="input_y"
        )

        # secondly, define embedding layers: [vocab_size, embedding_size]
        device = "/gpu:0" if self.gpu_flags else "/cpu:0"
        with tf.name_scope("embedding"), tf.device(device):
            # 共享变量
            embedding = tf.get_variable(
                name="embedding",
                shape=[self.vocabulary_size, self.embedding_size],
                dtype=tf.float32
            )
            # output shape: [batch_size, seq_length, embedding_size]
            embedding_outputs = tf.nn.embedding_lookup(
                params=embedding,
                ids=self.input_x,
                name="embedding_outputs"
            )
            # output shape: [batch_size, seq_length, embedding_size, in_channel=1]
            expand_embedding_outputs = tf.expand_dims(
                input=embedding_outputs,
                axis=-1,
                name="expand_embedding_outputs"
            )

        # thirdly, define conv-maxpool layers, to extract feature to generate feature maps
        # the num_filters means to extract num_filters feature maps
        # the filter_sizes, mean to convolution operation
        # 有多个不同的卷积大小的核
        maxpool_outputs = []
        for filter_size in self.filter_sizes:
            name = "conv_maxpool_{}".format(filter_size)
            with tf.name_scope(name):
                """
                    Computes a 2-D convolution given 4-D `input` and `filter` tensors.
                    Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
                    and a filter / kernel tensor of shape
                    `[filter_height, filter_width, in_channels, out_channels]`
                """
                # in_channel = 1
                filter_shape = [
                    filter_size, self.embedding_size,
                    1, self.num_filters
                ]
                conv_w = tf.Variable(
                    initial_value=tf.truncated_normal(filter_shape, stddev=0.1),
                    name="conv_weights_{}".format(filter_size),
                    dtype=tf.float32,
                )
                conv_b = tf.Variable(
                    initial_value=tf.constant(0.1, shape=[self.num_filters]),
                    name="conv_biases_{}".format(filter_size),
                    dtype=tf.float32,
                )

                # output shape: [batch, self.sequence_length-filetr_size+1, 1, self.num_filters]
                # 卷积操作ci = W*Xi:i+h-1
                conv_output = tf.nn.conv2d(
                    input=expand_embedding_outputs,
                    filter=conv_w,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="convolution_{}".format(filter_size)
                )
                # 偏置相加操作
                bias_added = tf.nn.bias_add(
                    value=conv_output,
                    bias=conv_b,
                    name="bias_added_{}".format(filter_size)
                )

                # 激活层操作fi = relu(ci + bi)
                relu_output = tf.nn.relu(
                    features=bias_added,
                    name="relu_output_{}".format(filter_size)
                )

                # 池化操作output shape:[batch, 1, 1, self.num_filters]
                maxpool_output = tf.nn.max_pool(
                    # value, 需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
                    # 依然是[batch, height, width, channels]这样的shape
                    value=relu_output,
                    # 池化窗口大小, [batch, height, width, channels]
                    # 一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化,
                    # 所以这两个维度设为了1
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    # 和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="maxpool_output_{}".format(filter_size)
                )

                # maxpool_outputs shape: [len(self.filter_sizes), batch_size, 1, 1, self.num_filters]
                maxpool_outputs.append(maxpool_output)

        # fourthly, define flatten layer: make the output of conv-maxool to shape [batch_size, ?, 1]
        num_filter_total = len(self.filter_sizes) * self.num_filters
        # shape => [batch_size, 1, 1, number_filter_total]
        maxpool_output_concat = tf.concat(
            values=maxpool_outputs,
            axis=-1
        )
        # shape: [batch_size, num_filter_total]
        flatten_layer_input = tf.reshape(
            tensor=maxpool_output_concat,
            shape=[-1, num_filter_total],
            name="flatten_layer_input"
        )

        # fifthly, define hidden layers: may have many layers
        # and set dropout_keep_probability to avoid overfitting
        # 这里也可以像上面一样手动定义w,b参数,搭建全连接层
        # shape: [batch_size, num_filter_total] => [batch_size, self.fc_hidden_size]
        with tf.name_scope(name="full_connected_layer"):
            fc_hidden = tf.layers.dense(
                inputs=flatten_layer_input,
                units=self.fc_hidden_size,
                name="fc_hidden_layer"
            )
            fc_dropout = tf.layers.dropout(
                inputs=fc_hidden,
                rate=self.keep_probability,
                name="fc_dropout"
            )
            fc_relu = tf.nn.relu(
                features=fc_dropout,
                name="fc_relu"
            )

        # sixthly, difine softmax layer, to predict the classes: shape [batch, labels_num]
        # shape: [batch_size, self.fc_hidden_size] => [batch_size, labels_num]
        with tf.name_scope(name="softmax_layer"):
            logits = tf.layers.dense(
                inputs=fc_relu,
                units=self.labels_num,
                name="fc_softmax"
            )
            self.y_pred_cls = tf.nn.softmax(logits=logits, name="y_softmax_predict")

        # seventhly, caculate the loss and
        # define the train optimization operation(backpropagation gradient algorithm)
        # to update the parameters,like w,b and so on
        with tf.name_scope(name="loss_optimize"):
            # 这个操作的输入logits是未经缩放的，该操作内部会对logits使用softmax操作
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.input_y,
                name="cross_entropy"
            )
            self.loss = tf.reduce_mean(cross_entropy)
            self.train_optimize = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            ).minimize(
                loss=self.loss,
                global_step=self.global_step_tensor,
                name="minimize_loss"
            )

        # eighthly, caculate the accuracy
        # shape: [batch_size, labels_num]
        with tf.name_scope(name="accuracy"):
            corrected_predict = tf.equal(
                x=tf.argmax(self.input_y, 1),
                y=tf.argmax(self.y_pred_cls, 1),
                name="label_equal"
            )
            self.accuracy = tf.reduce_mean(
                input_tensor=tf.cast(
                    x=corrected_predict,
                    dtype=tf.float32,
                ),
                name="caculate_accuracy"
            )

        print("搭建cnn模型完成!")
