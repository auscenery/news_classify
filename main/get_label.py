#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import json
import tensorflow as tf
import os
import tensorflow.contrib.keras as kr

from operator import itemgetter
from flask import Flask, request
from utils.access_control import access_control
from utils.misc import parse_args
from models.text_cnn import TextCnnModel
from preprocess.data import read_category, read_vocab, sub_not_chinese_word

log = logging.getLogger(__name__)
app = Flask(__name__)
configs = parse_args()
configs.model_name = 'best_validation-2400'
configs.gpu_flags = False
model = TextCnnModel(configs)
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# 注意模型的路径与名称, 需要从参数中传入进来
save_path = os.path.join(configs.model_dir, configs.model_name)
saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
words, word2ids = read_vocab(vocab_filename=configs.vocabulary_filename)
categories, cat2ids = read_category()


@app.route('/classify', methods=['POST'])
@access_control
def run():
    try:
        contents = request.form['content'].strip()
        body, x_pads = convert_body_to_ids(contents)
        feed_dict = {
            model.input_x: x_pads,
            model.keep_probability: 1,
        }
        y_pred_cls = session.run(
            fetches=model.y_pred_cls,
            feed_dict=feed_dict
        )
        y0 = [format(e, "<6.3f") for e in y_pred_cls[0]]
        l = [(categories[i], score) for i, score in enumerate(y0)]
        label_scores = sorted(l, key=itemgetter(1), reverse=True)
        pass
    except Exception as e:
        log.error(e)
        return json.dumps({
            'status': -1,
            'message': 'get label failed',
            'labels': {},
            'body': body
        })
    return json.dumps({
        'status': 0,
        'message': 'ok',
        'label_scores': label_scores,
        'body': body
    })


def convert_body_to_ids(contents, max_length=600):
    # body还需要预处理
    body = sub_not_chinese_word(contents)
    data_ids = [word2ids[x] for x in body if x in word2ids]
    # 使用keras提供的pad_sequences来将文本pad为固定长度, 在尾部添加
    x_pads = kr.preprocessing.sequence.pad_sequences(
        [data_ids], max_length, padding='post')

    return body, x_pads


if __name__ == "__main__":
    app.run()
