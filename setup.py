#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

print(find_packages())

setup(
    name="classify",
    version="1.0",
    description="news classify, data url:http://thuctc.thunlp.org"
                "/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews",
    author="moolighty",
    mail="moolighty@yahoo.com",
    license='MIT',
    packages=find_packages(),
    setup_requires=['nose>=1.0'],
    url='https://github.com/the-gigi/conman',
)