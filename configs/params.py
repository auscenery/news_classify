#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from utils.operate_file import create_dir

configs = {
    "embedding_size": 64,
    "sequence_length": 600,
    "labels_num": 10,
    "num_filters": 128,
    "filter_sizes": [2, 3, 4, 5],
    "vocabulary_size": 5774,
    "fc_hidden_size": 1024,
    "dropout_keep_prob": 0.5,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "num_epochs": 10,
    "print_per_batch": 100,
    "threshold_valid_evaluation_batch": 1000,
    "summary_save_per_batch": 10,
    "gpu_flags": True,
}

base_dir = os.path.realpath(os.path.abspath(__file__) + "/../..")
data_dir = "{}/data".format(base_dir)
vocabulary_filename = "{}/vocab/vocabulary.txt".format(data_dir)
train_filename = "{}/character/cnews.train.txt".format(data_dir)
val_filename = "{}/character/cnews.val.txt".format(data_dir)
test_filename = "{}/character/cnews.test.txt".format(data_dir)

model_dir = "{}/expriment/model".format(data_dir)
model_name = "best_validation"
tensorboard_dir = "{}/expriment/tensorboard".format(data_dir)
predict_dir = "{}/expriment/predict".format(data_dir)
create_dir(model_dir)
create_dir(tensorboard_dir)
create_dir(predict_dir)

configs['base_dir'] = base_dir
configs['data_dir'] = data_dir
configs['vocabulary_filename'] = vocabulary_filename
configs['train_filename'] = train_filename
configs['val_filename'] = val_filename
configs['test_filename'] = test_filename
configs['model_dir'] = model_dir
configs['model_name'] = model_name
configs['tensorboard_dir'] = tensorboard_dir
configs['predict_dir'] = predict_dir