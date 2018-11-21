#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import time

from configs.params import configs
from datetime import timedelta
from preprocess.data import batch_iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(model, x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_probability: keep_prob
    }
    return feed_dict


def evaluate(sess, model, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def parse_args():
    """
        add_argument:
            dest;action;type;choices;default;help;
            optional arguments: short or long;
            positional arguments; nargs
    """
    parser = argparse.ArgumentParser(description="set params and file path")
    parser.add_argument('-es', '--embedding_size', type=int, help="嵌入层维度",
                        default=configs['embedding_size'])
    parser.add_argument('-sl', '--sequence_length', type=int, help="文章序列长度",
                        default=configs['sequence_length'])
    parser.add_argument('-ln', '--labels_num', type=int, help="分类类别数目",
                        default=configs['labels_num'])
    parser.add_argument('-nf', '--num_filters', type=int, help="卷积核数据",
                        default=configs['num_filters'])
    parser.add_argument('-fs', '--filter_sizes', nargs='*', type=int, help="卷积核大小,可以有多个",
                        default=configs['filter_sizes'])
    parser.add_argument('-vs', '--vocabulary_size', type=int, help="字典大小",
                        default=configs['vocabulary_size'])
    parser.add_argument('-fhs', '--fc_hidden_size', type=int, help="全连接层的隐藏层维度大小",
                        default=configs['fc_hidden_size'])
    parser.add_argument('-dkp', '--dropout_keep_prob', type=int, help="全连接层的dropout保留概率大小",
                        default=configs['dropout_keep_prob'])
    parser.add_argument('-lr', '--learning_rate', type=float, help="梯度学习率的大小",
                        default=configs['learning_rate'])
    parser.add_argument('-bs', '--batch_size', type=int, help="数据训练的batch大小",
                        default=configs['batch_size'])
    parser.add_argument('-ne', '--num_epochs', type=int, help="训练周期大小",
                        default=configs['num_epochs'])
    parser.add_argument('-ppb', '--print_per_batch', type=int,
                        help="每多少轮对验证集合进行准确率计算检查并输出训练集和验证集合的情况,"
                             "并以此判断是否提前退出",
                        default=configs['print_per_batch'])
    parser.add_argument('-tveb', '--threshold_valid_evaluation_batch', type=int,
                        help="训练过程中当验证集合的准确率经过一定轮数后,还没有增长,退出的batch step阈值",
                        default=configs['threshold_valid_evaluation_batch'])
    parser.add_argument('-sspb', '--summary_save_per_batch', type=int,
                        help="每多少轮对当前的运行情况指标的保存到tensorboard目录下",
                        default=configs['summary_save_per_batch'])
    parser.add_argument('-gf', '--gpu_flags', type=int,
                        help="1使用gpu, 0还是使用cpu",
                        default=configs['gpu_flags'])
    parser.add_argument('-bd', '--base_dir', type=str,
                        help="项目路径",
                        default=configs['base_dir'])
    parser.add_argument('-dd', '--data_dir', type=str,
                        help="数据根路径",
                        default=configs['data_dir'])
    parser.add_argument('-vf', '--vocabulary_filename', type=str,
                        help="字典文件路径",
                        default=configs['vocabulary_filename'])
    parser.add_argument('-tf', '--train_filename', type=str,
                        help="训练文件的路径",
                        default=configs['train_filename'])
    parser.add_argument('-vaf', '--val_filename', type=str,
                        help="验证集合文件的路径",
                        default=configs['val_filename'])
    parser.add_argument('-tef', '--test_filename', type=str,
                        help="测试集数据文件的路径",
                        default=configs['test_filename'])
    parser.add_argument('-md', '--model_dir', type=str,
                        help="模型保存的路径",
                        default=configs['model_dir'])
    parser.add_argument('-mn', '--model_name', type=str,
                        help="模型保存的名字",
                        default=configs['model_name'])
    parser.add_argument('-td', '--tensorboard_dir', type=str,
                        help="tensorboard可视化收集指标保存的路径",
                        default=configs['tensorboard_dir'])
    parser.add_argument('-pd', '--predict_dir', type=str,
                        help="预测时生成的报表图片保存路径",
                        default=configs['predict_dir'])

    args = parser.parse_args()
    args.gpu_flags = True if args.gpu_flags != 0 else False

    return args

