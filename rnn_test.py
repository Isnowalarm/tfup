# -*- coding=utf-8 -*-
"""
#
# File Name: rnn_test.py
# @Author: @baidu.com
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 128)
print rnn_cell.state_size

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = 128)
print lstm_cell.state_size

inputs = tf.placeholder(np.float32, shape = (32, 100))
h0 = lstm_cell.zero_state(32, np.float32)
output, h1 = lstm_cell.call(inputs, h0)
print h1.h
print h1.c


