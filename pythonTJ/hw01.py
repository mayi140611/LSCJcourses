#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: hw01.py
@time: 2018/7/12 20:02
"""
from math import e


def sigmoid(x):
    return 1/(1+e**(-x))


if __name__ == '__main__':
    print(sigmoid(0))