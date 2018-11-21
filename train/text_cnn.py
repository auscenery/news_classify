#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import time
import tensorflow as tf

from preprocess.data import batch_iter
from preprocess.data import read_category, convert_example_to_ids, read_vocab
from utils.misc import get_time_dif, feed_data, evaluate


def run(session, model, configs):
    print("Configuring TensorBoard and Saver...")
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(configs.tensorboard_dir)

    # model Saver
    saver = tf.train.Saver()

    # 加载字典和类别
    print("loading dictionary and category labels infos:")
    start_time = time.time()
    words, word2ids = read_vocab(vocab_filename=configs.vocabulary_filename)
    categories, cat2ids = read_category()
    print("Time usage: {}".format(get_time_dif(start_time)))

    # 载入训练集与验证集
    print("Loading training and validation data...")
    start_time = time.time()
    # numpy array
    x_train, y_train = convert_example_to_ids(
        filename=configs.train_filename,
        word2ids=word2ids,
        cat2ids=cat2ids,
        max_length=model.sequence_length
    )
    x_val, y_val = convert_example_to_ids(
        filename=configs.val_filename,
        word2ids=word2ids,
        cat2ids=cat2ids,
        max_length=model.sequence_length
    )
    print("Time usage:{}", get_time_dif(start_time))

    # 这一步很重要, 否则程序运行保存,因为训练之前,变量一定要初始化,即赋值
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    # 开始训练
    print('Training and evaluating...')
    train_start_time = time.time()
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved_step = 0  # 记录上一次提升批次
    # 如果超过 require_improvement step 还未提升，提前结束训练
    require_improvement_steps = model.threshold_valid_evaluation_batch

    flag = False
    start_time = time.time()
    # batch step 自增器
    cur_step = model.global_step_tensor.eval(session)
    save_path = os.path.join(configs.model_dir, configs.model_name)
    for epoch in range(configs.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, model.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(model, x_batch, y_batch, model.dropout_keep_prob)

            if cur_step % configs.summary_save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(summary=s, global_step=cur_step)

            if cur_step % configs.print_per_batch == 0:
                # 计算当前的训练集合上的损失和正确率
                feed_dict[model.keep_probability] = 1.0
                loss_train, acc_train = session.run(
                    [model.loss, model.accuracy],
                    feed_dict=feed_dict
                )

                # 验证当前训练的结果在验证集合上的性能
                loss_val, acc_val = evaluate(session, model, x_val, y_val)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    saver.save(sess=session,
                               save_path=save_path,
                               global_step=cur_step)
                    best_acc_val = acc_val
                    last_improved_step = cur_step
                    improved_str = '**'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(cur_step, loss_train, acc_train,
                                 loss_val, acc_val, time_dif, improved_str))

            # 运行优化,更新相关参数w,b
            session.run(model.train_optimize, feed_dict=feed_dict)

            cur_step = model.global_step_tensor.eval(session)
            if cur_step - last_improved_step > require_improvement_steps:
                # 验证集正确率长期不提升，提前结束训练
                flag = True
                break  # 跳出循环

        if flag:  # 同上
            print("No optimization for a long time, auto-stopping in advance!")
            break

    train_total_times = get_time_dif(train_start_time)
    print("训练总共用时:{}".format(train_total_times))