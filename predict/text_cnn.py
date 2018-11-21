#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import time
import os
import tensorflow as tf
import numpy as np

from sklearn import metrics
from utils.misc import get_time_dif, evaluate
from preprocess.data import convert_example_to_ids, read_category, read_vocab
from utils.operate_csv import convert_all_to_csv, convert_diff_to_csv


def run(session, model, configs):
    print("Loading test data...")
    words, word2ids = read_vocab(configs.vocabulary_filename)
    categories, cat2ids = read_category()
    start_time = time.time()
    x_test, y_test = convert_example_to_ids(
        filename=configs.test_filename,
        word2ids=word2ids,
        cat2ids=cat2ids,
        max_length=configs.sequence_length
    )
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    # 注意模型的路径与名称, 需要从参数中传入进来
    save_path = os.path.join(configs.model_dir, configs.model_name)
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, model, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_probability: 1.0
        }
        # 注意维度对应
        y_pred_cls[start_id:end_id] = session.run(
            tf.argmax(model.y_pred_cls, 1), feed_dict=feed_dict
        )

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(
        y_test_cls, y_pred_cls, target_names=categories)
    )

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # from ipdb import set_trace
    # set_trace()

    # 保存结果到csv文件中
    predict_dir = configs.predict_dir
    all_csv_filename = "{}/all_csv_filename.csv".format(predict_dir)
    diff_csv_filename = "{}/diff_csv_filename.csv".format(predict_dir)
    convert_all_to_csv(x_test, y_test_cls, y_pred_cls, all_csv_filename, words, categories)
    convert_diff_to_csv(x_test, y_test_cls, y_pred_cls, diff_csv_filename, words, categories)
