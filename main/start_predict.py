#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf

from utils.misc import parse_args
from models.text_cnn import TextCnnModel
from predict.text_cnn import run

if __name__ == "__main__":
    # 超参数与路径设置
    configs = parse_args()

    # 生成model
    model = TextCnnModel(configs)

    # 生成会话并开始进行训练
    with tf.Session() as session:
        run(
            session=session,
            model=model,
            configs=configs
        )