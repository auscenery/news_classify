#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""编码与解码相关转化"""

import hashlib
import base64
import re

from urllib.parse import quote, unquote


def gbk2unicode(s):
    """
        gbk bytes to python unicode str
    """
    return s.decode(encoding='gbk', errors='ignore')


def unicode2gbk(s):
    """python unicode str to gbk bytes"""
    return s.encode(encoding='gbk', errors='ignore')


def utf82unicode(s):
    """utf-8 bytes to python unicode str"""
    return s.decode(encoding='utf-8', errors='ignore')


def unicode2utf8(s):
    """python unicode str to utf-8 bytes"""
    return s.encode(encoding='utf-8', errors='ignore')


def gbk2utf8(s):
    """gbk bytes to utf-8 bytes"""
    return s.decode(encoding='gbk', errors='ignore').encode('utf-8')


def utf82gbk(s):
    """utf-8 bytes to gbk bytes"""
    return s.decode(encoding='utf-8', errors='ignore').encode('gbk')


def get_md5(s, hex_flag=True):
    """
    得到md5值
    :param s:str, s必须是是unicode类型
    :param hex_flag: bool 是否返回十六进制字节串编码
    :return: md5 bytes, 默认十六进制
    """
    m = hashlib.md5()
    m.update(s.encode('utf-8', 'ignore'))
    return m.hexdigest() if hex_flag else m.digest()


def url_decode(s):
    """
    url解码
    """
    return unquote(s)


def url_encode(s):
    """
     url编码
    """
    return quote(s)


def base64_encode(s, encoding='utf-8'):
    if isinstance(s, str):
        return base64.b64encode(s.encode(encoding=encoding))
    if isinstance(s, (bytes, bytearray)):
        return base64.b64encode(s)
    return None


def base64_decode(s, encoding='utf-8'):
    if isinstance(s, (bytes, bytearray)):
        return base64.b64decode(s).decode(encoding=encoding)
    else:
        return None


def hanzi_to_pinyin(s, tone_flag=True):
    """将汉字转化为拼音, 默认带音调, 这里需要传入汉字"""
    from pypinyin import pinyin, lazy_pinyin
    s = s.strip()
    if tone_flag:
        return ' '.join([j for l in pinyin(s) for j in l])
    else:
        return ' '.join(lazy_pinyin(s))


def replace_characters(s, pattern, replace_str=''):
    """根据pattern替换成指定的字符"""
    return re.sub(pattern, replace_str, s)
