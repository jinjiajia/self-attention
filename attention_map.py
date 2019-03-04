'''打印出attention关注的词或字'''

from __future__ import print_function

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import jieba
import tensorflow.contrib.keras as kr


from cnn_model import TextCNN,TCNNConfig
from cnnw_model import TextCNNW,TCNNWConfig
from data.xlsx_loader import read_vocab



def stovect(config,message,model,words, word_to_id, type=True):
    content = message
    if type:
        data = [word_to_id[x] for x in content if x in word_to_id]
    else:
        data = [word_to_id[x] for x in jieba.lcut(content) if x in word_to_id]
    data = kr.preprocessing.sequence.pad_sequences([data], config.seq_length)
    datas= [words[int(x)] for x in data[0]]

    feed_dict = {
        model.input_x: data,
        model.input_istrain: -1
    }
    # print(kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length))###pad成固定长度
    fc = session.run(model.fc, feed_dict=feed_dict)
    return fc, datas


if __name__ == '__main__':
    ####char#####
    base_dir = 'data/cnews'
    vocab_dir = os.path.join(base_dir, 'vocab_cnews.txt')

    save_dir = 'checkpoints/modelattentioncnn'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
    config = TCNNConfig()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    data_xls = pd.read_excel("train_data18k.xlsx", usecols='B,H')  # skiprow=[0]去除第一行.usecols的第一列是B列，序号列不计入
    newdata = data_xls.values  # to numpy
    alltext = newdata[:, 0]
    f=open('charresult.txt', 'w', encoding='utf-8')
    for text in alltext:
        fc, data = stovect(config, text, model, words, word_to_id, type=True)
        fd = dict(zip(range(len(fc[0])), fc[0]))
        fa = sorted(fd.items(), key=lambda x: x[1], reverse=True)
        label = []
        for i in range(len(data)):
            if data[fa[i][0]] != '<PAD>':
                label.append(data[fa[i][0]])
        print(label)
        f.write(text+'\n'+str(label)+'\n')
    ####wordcnn#####
    # jieba.load_userdict("jiebauserdict100w.txt")
    # base_dir = 'data/cnews'
    # vocab_dir = os.path.join(base_dir, 'vocabword_cnews.txt')
    # save_dir = 'checkpoints/modelattentioncnnw'
    # save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
    #
    # config = TCNNWConfig()
    # words, word_to_id = read_vocab(vocab_dir)
    # config.vocab_size = len(words)
    # data = np.load('data/vector_word1.npz')
    # config.pre_trianing = data["embeddings"]
    # config.embedding_size = np.shape(config.pre_trianing)[1]
    # model = TextCNNW(config)
    # session = tf.Session()
    # session.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess=session, save_path=save_path)
    #
    # data_xls = pd.read_excel("train_data18k.xlsx", usecols='B,H')  # skiprow=[0]去除第一行.usecols的第一列是B列，序号列不计入
    # newdata = data_xls.values  # to numpy
    # alltext = newdata[:, 0]
    # f=open('wordresult.txt', 'w', encoding='utf-8')
    # for text in alltext:
    #     fc, data = stovect(config,text,model,words, word_to_id,type=False)
    #     fd = dict(zip(range(len(fc[0])), fc[0]))
    #     fa = sorted(fd.items(), key=lambda x: x[1], reverse=True)
    #     label = []
    #     for i in range(len(data)):
    #         if data[fa[i][0]] != '<PAD>':
    #             label.append(data[fa[i][0]])
    #     print(label)
    #     f.write(text+'\n'+str(label)+'\n')