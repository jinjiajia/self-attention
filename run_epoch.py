#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import random
from cnn_model import *
from cnnw_model import *
from data.xlsx_loader import *#enjoyer test

import os, codecs
import numpy as np
from datetime import timedelta


def export_word2vec_vectors(vocab, word2vec_dir, trimmed_filename):
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)
    print('saving npy word2vec')

def run_epoch(x_train, y_train, x_test, y_test, model, config):
    print('Constructing TensorFlow Graph...')
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    tensorboard_dir = 'tensorboard/model'+config.name
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    save_dir = 'checkpoints/model'+config.name
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # 配置 tensorboard
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)

    # 配置 Saver
    saver = tf.train.Saver()
    if os.path.exists(save_dir+'.index'):
        saver.restore(sess=session, save_path=save_dir)
        print("load and continue trainning")

    # 生成批次数据
    print('Generating batch...')
    batch_train = batch_iter(list(zip(x_train, y_train)),
        config.batch_size, config.num_epochs)
    print("batch_train", batch_train)

    def feed_data(batch,step,istraining):
        """准备需要喂入模型的数据"""
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.input_step: step,
            model.input_istrain: istraining

        }
        return feed_dict, len(x_batch)

    def evaluate(x_, y_):
        """
        模型评估
        一次运行所有的数据会OOM，所以需要分批和汇总
        """
        batch_eval = batch_iter(list(zip(x_, y_)), 128, 1)
        total_loss = 0.0
        total_acc = 0.0
        cnt = 0
        for batch in batch_eval:
            feed_dict, cur_batch_len = feed_data(batch,0,istraining=-1)
            loss, acc = session.run([model.loss, model.acc],
                feed_dict=feed_dict)
            total_loss += loss * cur_batch_len
            total_acc += acc * cur_batch_len
            cnt += cur_batch_len
        return total_loss / cnt, total_acc / cnt

    # 训练与验证
    print('Training and evaluating...')
    best_acc_val = 0.0  # 最佳验证集准确率
    print_per_batch = config.print_per_batch
    for i, batch in enumerate(batch_train):
        feed_dict, _ = feed_data(batch,i,istraining=config.istraining) 
        if i % 5 == 0:  # 每5次将训练结果写入tensorboard scalar
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, i)
        if i % print_per_batch == print_per_batch - 1:  # 每200次输出在训练集和验证集上的性能
            loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
            loss_val, acc_val = evaluate(x_test, y_test)
            if acc_val > best_acc_val:
                # 保存最好结果
                best_acc_val = acc_val
                saver.save(sess=session, save_path=save_path)
                improved_str = '*'
            else:
                improved_str = ''
            # msg = 'Iter: d%, Train Loss: f%, Train Acc: f%, Time: d%'
            print("Iter: %d, Train Loss: %f, Train Acc: %f, Val Loss: %f, Val Acc: %f"%((i + 1), loss_train, acc_train, loss_val, acc_val))
        session.run(model.optim, feed_dict=feed_dict)  # 运行优化

    # 最后在测试集上进行评估
    print('Evaluating on test set...')
    loss_test, acc_test= evaluate(x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    session.close()

if __name__ == '__main__':
###########cnnchar model
    print('Loading data...')
    trainlines,trainlab,testlines,testlab=wenti_read_file('train_data18k.xlsx',testsize=0.1)
    print('%d %d'%(len(trainlines),len(testlines)))
    vocab_path='data/cnews/vocab_cnews.txt'
    if not os.path.exists(vocab_path):
        build_vocab(trainlines)
    x_train, y_train, x_test, y_test, words = preocess_file(trainlines, trainlab, testlines, testlab)
    num_class = y_test.shape[1]
    print(num_class)
    print('Using CNN model...')
    config = TCNNConfig()
    config.vocab_size = len(words)
    config.num_classes = num_class
    model = TextCNN(config)
    run_epoch(x_train, y_train, x_test, y_test, model, config)
#########train cnnword, every model should be trainned seperately
    # jieba.load_userdict("jiebauserdict100w.txt")
    # trainlines,trainlab,testlines,testlab=wenti_read_file('train_data18k.xlsx',testsize=0.1)
    # print('%d %d'%(len(trainlines),len(testlines)))
    # vocab_path='data/cnews/vocabword_cnews.txt'
    # if not os.path.exists(vocab_path):
    #     build_vocabword(trainlines)
    # x_train, y_train, x_test, y_test, words = preocess_wordfile(trainlines, trainlab, testlines, testlab)
    # num_class = y_test.shape[1]
    # print(num_class)
    # print('Using CNN model...')
    # config = TCNNWConfig()
    # config.vocab_size = len(words)
    # config.num_classes = num_class
    # if not os.path.exists(config.vector_word_npz):
    #     export_word2vec_vectors(vocab_path, config.word2vec_dir, config.vector_word_npz)
    # data = np.load('data/vector_word1.npz')
    # config.pre_trianing = data["embeddings"]
    # config.embedding_size=np.shape(config.pre_trianing)[1]
    # model = TextCNNW(config)
    # run_epoch(x_train, y_train, x_test, y_test, model, config)

