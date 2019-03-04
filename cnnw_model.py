#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from attention_model import *

class TCNNWConfig(object):
    """CNN配置参数"""
    # 模型参数
    embedding_size = 200      # 词向量维度
    seq_length = 100        # 序列长度
    num_classes =85      # 类别数
    num_filters = 256     # 卷积核数目
    kernel_size = 2        # 卷积核尺寸###word 2 char 3
    vocab_size = 12000      # 词汇表达小
    pre_trianing = None
    hidden_dim = 128     # 全连接层神经元

    dropout_keep_prob = 0.8 # dropout保留比例训练0.5
    learning_rate = 1e-3   # 学习率
    istraining=1
    input_step=1
    batch_size = 512     # 每批训练大小
    num_epochs = 70     # 总迭代轮次
    name='attentioncnnw'
    print_per_batch = 200    # 每多少轮输出一次结果

    nb_head = 8
    num_blocks = 6
    word2vec_dir= './data/100W-word2vec.txt' ###  get from svn
    vector_word_npz = './data/vector_word1.npz'

class TextCNNW(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config
        # tf.reset_default_graph()#清空所有模型数据，重新建模型。
        self.input_x = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.num_classes], name='input_y')
        self.input_step = tf.placeholder(tf.int32,name='input_step')
        self.input_istrain=tf.placeholder(tf.int32,name='istraining')
        self.cnn()

    def input_embedding(self):
        """词嵌入"""
        with tf.device('/cpu:0'):
            wordembedding = tf.get_variable("wordembeddings",
                                            shape=[self.config.vocab_size, self.config.embedding_size],
                                            initializer=tf.constant_initializer(self.config.pre_trianing))
            _inputs = tf.nn.embedding_lookup(wordembedding, self.input_x)
        return _inputs

    def attention(self):
        embedding_inputs = self.input_embedding()
        with tf.variable_scope('add_Position_Embedding'):
            position_embedding=positional_encoding(self.input_x,self.config.embedding_size)
            position_embedding = tf.add(position_embedding, embedding_inputs)

        with tf.variable_scope('dropout-embeddings'):
            position_embedding = tf.layers.dropout(position_embedding,
                                         rate=0,
                                         training=tf.convert_to_tensor(True))

        with tf.variable_scope("multihead_attention"):
            ## Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc = multihead_attention(keys=position_embedding,
                                              num_units=self.config.embedding_size,
                                              num_heads=self.config.nb_head,
                                              dropout_rate=1-self.config.dropout_keep_prob,
                                              is_training=True)
                    enc = feedforward(enc, num_units=self.config.embedding_size)

        return enc


    def cnn(self):
        """cnn模型"""
        enc = self.attention()
        # conv = tf.layers.conv1d(enc,
        #         self.config.num_filters,
        #         self.config.kernel_size, name='conv1')
        gmp = tf.reduce_max(enc, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.seq_length, name='fc1')
            fc=tf.cond(self.input_istrain>0,lambda :tf.contrib.layers.dropout(fc,self.config.dropout_keep_prob),
                       lambda :tf.contrib.layers.dropout(fc,1.0))
            fc = tf.nn.relu(fc)

            # 分类器
            self.fc=fc
            self.logits = tf.layers.dense(fc, self.config.num_classes,
                name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name='fc3')  # 预测类别

        with tf.name_scope("loss"):#todo
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("optimize"):
            # 优化器
            decay=pow(0.9,self.input_step/100)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate*decay)
            self.optim = optimizer.minimize(self.loss)
        #
        with tf.name_scope("accuracy"):#todo
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)  ##预测分类的
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
