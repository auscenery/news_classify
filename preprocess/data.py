#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import re
import os
import numpy as np
import tensorflow.contrib.keras as kr

from collections import Counter
from utils.operate_file import get_files_recursive


def sub_eng_digit_other(content):
    """替换掉英文,数字,其他等等"""
    engligh_pattern = "[a-zA-Z]+"
    content1 = re.sub(string=content, repl="<ENG>", pattern=engligh_pattern, flags=re.S | re.M).strip()
    other_pattern = "[^a-zA-Z0-9\u4E00-\u9FA5<>]+"
    content1 = re.sub(string=content1, repl="<OTHER>", pattern=other_pattern, flags=re.S | re.M).strip()
    digit_pattern = "[0-9]+"
    content1 = re.sub(string=content1, repl="<NUM>", pattern=digit_pattern, flags=re.S | re.M).strip()
    results = []
    s = ""
    for word in content1:
        if "\u4E00" <= word <= "\u9FA5":
            results.append(word)
        elif word == ">":
            if s.startswith("<"):
                results.append(s[1:])
            s = ""
        else:
            s += word
    return " ".join(results)


def sub_not_chinese_word(content):
    chinese_patten = "[\u4E00-\u9FA5]+"
    l = re.findall(pattern=chinese_patten, string=content, flags=re.S|re.M)
    results = [word.strip() for l1 in l for word in l1 if word.strip()]
    return " ".join(results)


def read_labels_contents(filename, callback, delimiter="\t", encoding='utf-8'):
    with open(filename, "r", encoding=encoding) as fp:
        results = []
        for line in fp.readlines():
            label, content = line.split(delimiter)
            results.append((label, callback(content)))
    return results


def write_to_txt_files(results, filename, encoding='utf-8'):
    with open(filename, 'w+', encoding=encoding) as fp:
        content = "\n".join([",".join(e) for e in results])
        fp.write(content)


def build_vocab(file_path_or_name, vocab_filename, vocab_size=-1,
                encoding='utf-8', delimiter=","):
    """
        根据训练集构建词汇表，存储; vocab_size等于-1表示用所有的字来构造词典
    """
    if os.path.isfile(file_path_or_name):
        files = [file_path_or_name]
    else:
        files = get_files_recursive(file_path_or_name)

    all_data = []
    max_word_length = 0
    for file in files:
        with open(file, 'r', encoding=encoding) as fp:
            lines = fp.readlines()
            for line in lines:
                label, content = line.split(delimiter)
                words = content.strip().split(" ")
                n = len(words)
                if max_word_length < n:
                    max_word_length = n
                all_data.extend(words)
        print("文件{}的所有样本中,最大样本单词长度为:{}".format(file, n))

    counter = Counter(all_data)
    if vocab_size == -1:
        vocab_size = len(counter)
    count_pairs = counter.most_common(vocab_size)
    words, _ = list(zip(*count_pairs))
    # 添加一个 PAD 来将所有文本pad为同一长度
    words = ['PAD'] + list(words)
    with open(vocab_filename, mode='w+', encoding='utf-8') as fp:
        fp.write('\n'.join(words))


def read_vocab(vocab_filename, encoding="utf-8"):
    """读取词汇表"""
    with open(vocab_filename, 'r', encoding=encoding) as fp:
        words = [w.strip() for w in fp.readlines() if w.strip()]
    word2ids = dict(zip(words, range(len(words))))
    return words, word2ids


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    cat2ids = dict(zip(categories, range(len(categories))))

    return categories, cat2ids


def to_words(ids, vocabs, delimiter=""):
    """将id表示的内容转换为文字"""
    return delimiter.join(vocabs[x] for x in ids)


def convert_example_to_ids(filename, word2ids, cat2ids, delimiter=",", max_length=600, encoding='utf-8'):
    """将所有的文件转换为id表示, return numpy array tuple"""
    contents = []
    labels = []
    with open(filename, 'r', encoding=encoding) as fp:
        for line in fp.readlines():
            label, content = line.split(delimiter)
            contents.append(content.strip())
            labels.append(label.strip())

    data_ids, label_ids = [], []
    for i in range(len(contents)):
        data_ids.append([word2ids[x] for x in contents[i] if x in word2ids])
        label_ids.append(cat2ids[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度, 在尾部添加
    x_pads = kr.preprocessing.sequence.pad_sequences(data_ids, max_length, padding='post')

    # 将标签转换为one-hot表示
    y_pads = kr.utils.to_categorical(label_ids, num_classes=len(cat2ids))

    return x_pads, y_pads.astype('int')


def batch_iter(x, y, batch_size=64):
    """随机生成批次数据, x,y 均为numpy narray对象,对应正文,标签"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    # 重新随机排列
    indices = np.random.permutation(len(x))
    x_shuffles = x[indices]
    y_shuffles = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        x_ = x_shuffles[start_id:end_id]
        y_ = y_shuffles[start_id:end_id]
        yield x_, y_


if __name__ == "__main__":
    from pprint import pprint
    file_path = os.path.realpath("..")
    data_path = "{}/data".format(file_path)
    character_path = "{}/character".format(data_path)
    origin_path = "{}/original".format(data_path)
    filename_list = ["cnews.val.txt", "cnews.test.txt", "cnews.train.txt"]
    #for filename in filename_list:
    #    original_filename = "{}/{}".format(origin_path, filename)
    #    results = read_labels_contents(original_filename, sub_not_chinese_word)
    #    character_val_filename = "{}/{}".format(character_path, filename)
    #    write_to_txt_files(results, character_val_filename)
    vocab_filename = "{}/vocab/vocabulary.txt".format(data_path)
    #build_vocab(character_path, vocab_filename)
    words, word2ids = read_vocab(vocab_filename)
    #id2words_filename = "{}/vocab/id2words.txt".format(data_path)
    #words2id_filename = "{}/vocab/words2id.txt".format(data_path)
    #with open(id2words_filename, "w+", encoding="utf-8") as fp:
    #    fp.write("\n".join(words))
    #temp_word2ids = ["{}:{}".format(k, v) for k, v in word2ids.items()]
    #with open(words2id_filename, "w+", encoding="utf-8") as fp:
    #    fp.write("\n".join(temp_word2ids))
    categories, cat2ids = read_category()
    for file in filename_list:
        filename = "{}/{}".format(character_path, file)
        x_pads, y_pads = convert_example_to_ids(filename, word2ids, cat2ids)
        #results = [
        #    "{},{}".format(
        #        " ".join([str(y) for y in y_pads[i].tolist()]),
        #        " ".join([str(x) for x in x_pads[i].tolist()])
        #    ) for i in range(len(x_pads))
        #]
        #temp_filename = "{}/ids/{}".format(character_path, file)
        #with open(temp_filename, 'w+', encoding="utf-8") as fp:
        #    fp.write("\n".join(results))
        batches = batch_iter(x_pads, y_pads)
        for x, y in batches:
            print(x.tolist())
            print(y.tolist())

