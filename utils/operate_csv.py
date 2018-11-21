#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np


def convert_all_to_csv(x_test, y_test, y_predict, csv_filename, words, categories):
    x_test_temp = [to_words(ids, words) for ids in x_test]
    y_test_temp = [categories[i] for i in y_test]
    y_predict_temp = [categories[i] for i in y_predict]
    d = {
        "actual_label": y_test_temp,
        "predict_label": y_predict_temp,
        "body": x_test_temp
    }
    df = pd.DataFrame(d)
    df.to_csv(csv_filename)


def convert_diff_to_csv(x_test, y_test, y_predict, csv_filename, words, categories):
    n = len(x_test)
    indices = [i for i in range(n) if y_test[i] != y_predict[i]]
    convert_all_to_csv(
        x_test[indices], y_test[indices], y_predict[indices],
        csv_filename, words, categories
    )


def to_words(ids, vocabs, delimiter=""):
    """将id表示的内容转换为文字"""
    return delimiter.join(vocabs[x] for x in ids)

