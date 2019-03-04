#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow.contrib.keras as kr
import numpy as np
import os,re
import pandas as pd
import jieba

# jieba.load_userdict("jiebauserdict100w.txt")

def wenti_read_file(filename,testsize=0.1):
    """读取文件数据"""
    data_xls=pd.read_excel(filename,usecols='B,H')#skiprow=[0]去除第一行.usecols的第一列是B列，序号列不计入
    newdata=data_xls.values#to numpy
    alltext=newdata[:,0]
    alllab=np.array(newdata[:,1],np.int16)#dtype from object to int16
    # lineswords=[]
    # for i in range(len(alltext)):
    #     lineswords.append(' '.join(jieba.cut(alltext[i])))
    trainlines,testlines,trainlab,testlab=train_test_split(alltext,alllab,test_size=testsize)
    assert(len(trainlab)==len(trainlines))
    return trainlines,trainlab,testlines,testlab

def build_vocab(data, vocab_size=50000):
    """根据训练集构建词汇表，存储，使用常用字做词典"""
    all_data = []
    for content in data:
        all_data.extend(content)#变成单字，why
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open('data/cnews/vocab_cnews.txt', 'w',
        encoding='utf-8').write('\n'.join(words))


def build_vocabword(data, vocab_size=12000):
    """根据训练集构建词汇表，存储，使用常用词做词典"""
    all_data = []
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z+#&_%]+)")
    for line in data:
        try:
            line = line.rstrip()
            blocks = re_han.split(line)
            word = []
            for blk in blocks:
                if re_han.match(blk):
                    word.extend(jieba.lcut(blk))
            bwords = []
            for wd in word:
                if len(wd) > 1:
                    bwords.append(wd)
            all_data.extend(bwords)
        except:
            pass
    print('read over')
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open('data/cnews/vocabword_cnews.txt', 'w',
        encoding='utf-8').write('\n'.join(words))


def read_vocab(filename):
    """读取词汇表"""
    words = list(map(lambda line: line.strip(),
        open(filename, 'r', encoding='utf-8').readlines()))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


def _fileword_to_ids(sentences,labs,word_to_id,max_length=100):
    ##词级
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&_%]+)")  # the method of cutting text by punctuation
    contents=[]
    for line in sentences:
        try:
            line=line.rstrip()
            blocks = re_han.split(line)
            word = []
            for blk in blocks:
                if re_han.match(blk):
                    word.extend(jieba.lcut(blk))
            contents.append(word)
        except:
            pass
    data_id=[]
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(labs)
    return x_pad, y_pad

def _file_to_ids(contents,labels, word_to_id, max_length=200):
    ##字级
    """将文件转换为id表示"""
    data_id = []
    label_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(labels[i])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
    return x_pad, y_pad


def preocess_wordfile(trainlines,trainlab,testlines,testlab,seq_length=100):
    ####词级
    """一次性返回所有数据"""
    data_path='data/cnews'
    words, word_to_id = read_vocab(os.path.join(data_path,'vocabword_cnews.txt'))
    x_train,y_train=_fileword_to_ids(trainlines,trainlab, word_to_id, seq_length)
    x_test,y_test=_fileword_to_ids(testlines,testlab, word_to_id, seq_length)

    return x_train, y_train, x_test, y_test, words


def preocess_file(trainlines,trainlab,testlines,testlab,seq_length=200):
    """一次性返回所有数据"""
    data_path='data/cnews'
    words, word_to_id = read_vocab(os.path.join(data_path,'vocab_cnews.txt'))

    x_train,y_train=_file_to_ids(trainlines,trainlab, word_to_id, seq_length)
    x_test,y_test=_file_to_ids(testlines,testlab, word_to_id, seq_length)

    return x_train, y_train, x_test, y_test, words


def batch_iter(data, batch_size=64, num_epochs=5):
    """生成批次数据"""
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[indices]

        for batch_num in range(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    trainlines,trainlab,testlines,testlab=wenti_read_file('train_data18k.xlsx')
    if not os.path.exists('data/cnews/vocab_cnews.txt'):
        build_vocab(trainlines)





