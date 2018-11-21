#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""文件目录相关操作"""

import os
import logging
import json
import shutil

log = logging.getLogger(__name__)


def exist_file(file_path):
    try:
        return True if os.path.exists(file_path) else False
    except Exception as e:
        log.error("判断文件:{} 发生异常, 原因为:{}".format(file_path, e))
    return False


def get_dir_files(path):
    l = os.listdir(path)
    if not path.endswith('/'):
        path += '/'
    # 去掉隐藏文件
    return [path + file for file in l if not file.startswith('.')]


def create_dir(path):
    try:
        path = path.strip()
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        logging.exception(e)

    if os.path.exists(path):
        # log.info('{}文件夹创建成功或者已经存在'.format(path))
        return True
    else:
        return False


def write_item_to_file(file_path, item, encoding='utf-8', mode='w+'):
    """将item写入文件中"""
    if not item:
        log.warning("item为空,不写入文件{}中".format(file_path))
        return False
    if not file_path:
        log.warning("file_name:{}非法, 不进行写入".format(file_path))
        return False

    dir_name = os.path.dirname(file_path)
    if not create_dir(dir_name):
        log.warning("创建文件夹:{}失败, 不写入item".format(dir_name))

    with open(file=file_path, mode=mode, encoding=encoding) as fp:
        try:
            if isinstance(item, str):
                fp.write(item)
            else:
                s = json.dumps(obj=item, ensure_ascii=False)
                fp.write(s)

            if exist_file(file_path):
                return True
        except Exception as e:
            logging.exception(e)
    return False


def get_files_recursive(path, files=[]):
    # 得到所有的文件,排除目录,如果是目录,就递归处理
    file_dirs = get_dir_files(path)
    for fd in file_dirs:
        if os.path.isdir(fd):
            get_files_recursive(fd, files)
        else:
            files.append(fd)
    return list(set(files))


def read_file_content_bytes(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as fp:
            return fp.read().encode(encoding=encoding)
    except Exception as e:
        logging.exception(e)
    return None


def read_file_content_str(file_path, encoding='utf-8'):
    content = read_file_content_bytes(file_path)
    if isinstance(content, (bytes, bytearray)):
        return content.decode(encoding=encoding)
    else:
        log.error("读取文件:{} 失败".format(file_path))
        return None


def move_file_or_dir_to_dst(src, dest_dir):
    """移动文件或者目录到指定的目录"""
    try:
        if not exist_file(src):
            log.warning("文件或者目录:{}, 不存在,无法进行移动!".format(src))
            return False
        if create_dir(dest_dir):
            file_path = shutil.move(src, dest_dir)
            if file_path:
                return exist_file(file_path)
    except Exception as e:
        logging.exception(e)
    return False
