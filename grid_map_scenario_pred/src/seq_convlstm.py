from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
#import cv2
import scipy.misc as misc
import scipy.sparse
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
import pickle
import shutil
import tensorflow as tf
import glob
import random
import time
import datetime
import math
from configuration import Configuration

tf.logging.set_verbosity(tf.logging.INFO)

#DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_10frames_1ts/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/seq_convlstm_model_1ts/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/seq_convlstm_model_1ts_1/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/seq_convlstm_model_1ts_2/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/seq_convlstm_model_1ts_bd_att/"

#for 2ts
DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_10frames_2ts/"
MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/seq_convlstm_model_2ts/"

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0001
TRAINING_STEPS = 1000
THRESHOLD_ARRAYS = []
POOL_SIZE  = 2
KERNEL_SIZE = 3
PADDING_SAME = "same"
PADDING_VALID = "valid"
LABELS_PIXEL_1_COUNT = 0
TRAIN_SET = []
VALIDATION_SET = []
TEST_SET = []
eps = 1e-10
NUM_CLASSES = 2 # {0 = background, 1 = car}
NO_OF_FRAMES = 5
INPUT_HEIGTH = 450
INPUT_WIDTH = 100
FILTERS = [8, 16, 32, 64, 128, 256, 512]
RES_FILTERS = [32, 64, 128, 256, 512, 1024]
ALPHA = 0.5
BETA = 0.95

'''
# 1 time step ahead frame prediction
TOTAL_SAMPLES = 4704
TRAIN_SAMPLES = 4000
VALIDATION_SAMPLES = 600
TEST_SAMPLES = 104
'''

# 2 time step ahead frame prediction
TOTAL_SAMPLES = 4596
TRAIN_SAMPLES = 4000
VALIDATION_SAMPLES = 500
TEST_SAMPLES = 96

'''
# 3 time step ahead frame prediction
TOTAL_SAMPLES = 4704
TRAIN_SAMPLES = 4000
VALIDATION_SAMPLES = 500
TEST_SAMPLES = 204
'''

def reshape_input(x):
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    for i in range(BATCH_SIZE):
        #print('i elem>:', str(i))
        #print(input_arr[i][0])
        x1.append(x[i][0])
        x2.append(x[i][1])
        x3.append(x[i][2])
        x4.append(x[i][3])
        x5.append(x[i][4])
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    x4 = np.asarray(x4)
    x5 = np.asarray(x5)
    #print(x1.shape)
    x1 = np.reshape(x1, [x1.shape[0], x1.shape[1], x1.shape[2], 1])
    #print(x1.shape)
    x2 = np.reshape(x2, [x2.shape[0], x2.shape[1], x2.shape[2], 1])
    #print(x2.shape)
    x3 = np.reshape(x3, [x3.shape[0], x3.shape[1], x3.shape[2], 1])
    #print(x3.shape)
    x4 = np.reshape(x4, [x4.shape[0], x4.shape[1], x4.shape[2], 1])
    #print(x3.shape)
    x5 = np.reshape(x5, [x5.shape[0], x5.shape[1], x5.shape[2], 1])
    #print(x3.shape)

    return x1, x2, x3, x4, x5

def conv2d_layer(input, channel, kernel, stride, pad):
    conv2d = tf.layers.conv2d(inputs=input, filters=channel, kernel_size=[kernel, kernel], strides=(stride, stride), padding=pad)
    #conv2d = tf.layers.conv2d(inputs=conv2d, filters=channel, kernel_size=[kernel, kernel], strides=(stride, stride), padding=pad)
    conv2d = tf.layers.batch_normalization(conv2d, axis=-1, training=True)
    conv2d = tf.nn.relu(conv2d, name='conv2d_relu')
    #conv2d = tf.nn.elu(conv2d, name='conv2d_elu')
    return conv2d

def _conv_block(input, filters, kernel, stride, pad):
    x = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel, strides=stride, padding=pad)
    x = tf.layers.batch_normalization(x, axis=-1, training=True)
    x = tf.nn.relu(x)
    #x = tf.nn.elu(x)
    return x

def start_conv_block(input, filters, kernel, stride, input_cnt):
    x = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=[kernel, kernel], strides=(stride, stride), padding='same', name='conv1_'+ input_cnt)
    #x = tf.nn.elu(x, name='conv1_'+ input_cnt +'_elu')
    x = tf.layers.batch_normalization(x, axis=-1, training=True, name='conv1_'+ input_cnt +'_bn')
    x = tf.nn.relu(x, name='conv1_'+ input_cnt +'_relu')
    return x

def start_pool_block(input, kernel, stride, input_cnt):
    x = tf.layers.max_pooling2d(inputs=input, pool_size=kernel, strides=stride, padding='valid', name='pool1_'+ input_cnt)
    return x

def _pool_block(self, input, kernel, stride, pad):
    x = tf.layers.max_pooling2d(inputs=input, pool_size=[kernel, kernel], strides=stride, padding=pad)
    return x

def conv_res_block(input, filters, kernel, stride, block, identity_connection=True):
    filter1 = filters
    filter2 = filters * 4
    conv_name_base = 'res_' + block + '_branch'
    bn_name_base = 'bn_' + block + '_branch'

    #branch1
    if not identity_connection:
        if block == 'x1_3a' or block == 'x2_3a' or block == 'x3_3a' or block == 'x4_3a' or block == 'x5_3a':
            shortcut = tf.layers.conv2d(inputs=input, filters=filter2, kernel_size=[1, 2], strides=(stride, stride), padding='valid', name=conv_name_base+'1')
        elif block == 'x1_4a' or block == 'x2_4a' or block == 'x3_4a' or block == 'x4_4a' or block == 'x5_4a':
            shortcut = tf.layers.conv2d(inputs=input, filters=filter2, kernel_size=[1, 1], strides=(stride, stride), padding='valid', name=conv_name_base+'1')
        else:
            shortcut = tf.layers.conv2d(inputs=input, filters=filter2, kernel_size=[1, 1], strides=(stride, stride), padding='same', name=conv_name_base+'1')
        shortcut = tf.layers.batch_normalization(shortcut, axis=-1, training=True, name=bn_name_base+'1')
    else:
        shortcut = input

    #branch2
    if block == 'x1_3a' or block == 'x2_3a' or block == 'x3_3a' or block == 'x4_3a' or block == 'x5_3a':
        x = tf.layers.conv2d(inputs=input, filters=filter1, kernel_size=[1, 2], strides=(stride, stride), padding='valid', name=conv_name_base+'2a')
    elif block == 'x1_4a' or block == 'x2_4a' or block == 'x3_4a' or block == 'x4_4a' or block == 'x5_4a':
        x = tf.layers.conv2d(inputs=input, filters=filter1, kernel_size=[1, 1], strides=(stride, stride), padding='valid', name=conv_name_base+'2a')
    else:
        x = tf.layers.conv2d(inputs=input, filters=filter1, kernel_size=[1, 1], strides=(stride, stride), padding='same', name=conv_name_base+'2a')
    x = tf.layers.batch_normalization(x, axis=-1, training=True, name=bn_name_base+'2a')
    x = tf.nn.relu(x, name=conv_name_base+'2a_relu')
    #x = tf.nn.elu(x, name=conv_name_base+'2a_elu')


    x = tf.layers.conv2d(inputs=x, filters=filter1, kernel_size=[kernel, kernel], strides=(1, 1), padding='same', name=conv_name_base+'2b')
    x = tf.layers.batch_normalization(x, axis=-1, training=True, name=bn_name_base+'2b')
    x = tf.nn.relu(x, name=conv_name_base+'2b_relu')
    #x = tf.nn.elu(x, name=conv_name_base+'2b_elu')

    x = tf.layers.conv2d(inputs=x, filters=filter2, kernel_size=[1, 1], strides=(1, 1), padding='same', name=conv_name_base+'2c')
    x = tf.layers.batch_normalization(x, axis=-1, training=True, name=bn_name_base+'2c')

    #add
    x = tf.add(x, shortcut, name='res' + block)
    #x = tf.nn.elu(x, name='res' + block + '_elu')
    x = tf.nn.relu(x, name='res' + block + '_relu')
    return x

def skip_convlstm_block(input_shape, input_stack, kernel, init_state, name):
    #print('input_shape: ', input_shape)
    stack_frames = tf.stack([input_stack], axis=1)
    #print('stack_frames: ', stack_frames)
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[input_shape[0], input_shape[1], input_shape[2]],
                output_channels=input_shape[2],
                kernel_shape=[kernel, kernel],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name=name)

    if init_state == None:
        init_state = convlstm_layer.zero_state(BATCH_SIZE, dtype=tf.float32)

    #print('init_state: ', init_state)
    convlstm_output, hidden_state = tf.nn.dynamic_rnn(convlstm_layer, stack_frames, initial_state=init_state, dtype="float32")
    #print('convlstm_output: ', convlstm_output)
    #print('hidden_state :', hidden_state)

    convlstm_output = tf.split(convlstm_output, tf.ones((1), dtype=tf.int32 ), 1)
    #print('convlstm_output: ', convlstm_output)
    convlstm_output_last = convlstm_output[-1]
    convlstm_output_last = tf.reshape(convlstm_output_last, [BATCH_SIZE, input_shape[0], input_shape[1], input_shape[2]])
    #print('convlstm_output_last: ', convlstm_output_last)
    return convlstm_output_last, hidden_state

def dec_convlstm_block(input_shape, input_stack, kernel, init_state, name):
    #print('input_shape: ', input_shape)
    #with tf.variable_scope(name):
    stack_frames = tf.stack([input_stack], axis=1)
    #print('stack_frames: ', stack_frames)
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[input_shape[0], input_shape[1], input_shape[2]],
                output_channels=input_shape[2],
                kernel_shape=[kernel, kernel],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name=name)

    if init_state == None:
        init_state = convlstm_layer.zero_state(BATCH_SIZE, dtype=tf.float32)

    #print('init_state: ', init_state)
    convlstm_output, hidden_state = tf.nn.dynamic_rnn(convlstm_layer, stack_frames, initial_state=init_state, dtype="float32")
    #print('convlstm_output: ', convlstm_output)
    #print('hidden_state :', hidden_state)

    convlstm_output = tf.split(convlstm_output, tf.ones((1), dtype=tf.int32 ), 1)
    #print('convlstm_output: ', convlstm_output)
    convlstm_output_last = convlstm_output[-1]
    convlstm_output_last = tf.reshape(convlstm_output_last, [BATCH_SIZE, input_shape[0], input_shape[1], input_shape[2]])
    #print('convlstm_output_last: ', convlstm_output_last)
    return convlstm_output_last, hidden_state

def convlstm_block(input_shape, input_stack, kernel, init_state, name):
    #with tf.variable_scope(name):
    stack_frames = tf.stack([input_stack[0], input_stack[1], input_stack[2], input_stack[3], input_stack[4]], axis=1)
    print('stack_frames: ', stack_frames)
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[input_shape[0], input_shape[1], input_shape[2]],
                output_channels=input_shape[2],
                kernel_shape=[kernel, kernel],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name=name)

    if init_state == None:
        init_state = convlstm_layer.zero_state(BATCH_SIZE, dtype=tf.float32)

    #print('init_state: ', init_state)
    convlstm_output, hidden_state = tf.nn.dynamic_rnn(convlstm_layer, stack_frames, initial_state=init_state, dtype="float32")
    #print('convlstm_output: ', convlstm_output)
    #print('hidden_state :', hidden_state)

    convlstm_output = tf.split(convlstm_output, tf.ones((NO_OF_FRAMES), dtype=tf.int32 ), 1)
    #print('convlstm_output: ', convlstm_output)
    convlstm_output_last = convlstm_output[-1]
    convlstm_output_last = tf.reshape(convlstm_output_last, [BATCH_SIZE, input_shape[0], input_shape[1], input_shape[2]])
    #print('convlstm_output_last: ', convlstm_output_last)
    return convlstm_output_last, hidden_state
'''
def encoder_conv_net(x, mode):
    with tf.device('/gpu:0'):
        #config = tf.ConfigProto(allow_soft_placement=True)
        training = tf.equal(mode, 'train')
        inp_cnt = len(x)
        # ResNet101 as a backbone structure for encoder
        # strating block
        # 1st conv + relu + bn + convlstm block
        conv_stack1 = []
        for i in range(0, inp_cnt):
            conv1_x = start_conv_block(x[i], FILTERS[1], KERNEL_SIZE, 2, str(i+1))  #[batch_size, 225, 50, 16]
            print('conv1_x: ', conv1_x.shape)
            conv_stack1.append(conv1_x)

        pool1_stack = []
        for j in range(0, inp_cnt):
            pool1_x = start_pool_block(conv_stack1[j], (KERNEL_SIZE, 2), (2,2), str(j+1))  #[batch_size, 112, 25, 16]
            print('pool1_x: ', pool1_x.shape)
            pool1_stack.append(pool1_x)

        # residual block1
        conv_stack2 = []
        for k in range(0, inp_cnt):
            frame = k+1
            conv2_x = conv_res_block(pool1_stack[k], FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frame)+'_2a', identity_connection=False) # [batch_size, 112, 25, 64]
            conv2_x = conv_res_block(conv2_x, FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frame)+'_2b')
            conv2_x = conv_res_block(conv2_x, FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frame)+'_2c')
            print("BLOCK1 - X" + str(frame)+": ", conv2_x.shape)
            conv_stack2.append(conv2_x)

        # residual block2
        conv_stack3 = []
        for l in range(0, inp_cnt):
            frame = l+1
            conv3_x = conv_res_block(conv_stack2[l], FILTERS[2], KERNEL_SIZE, 2, 'x'+str(frame)+'_3a', identity_connection=False) # [batch_size, 56, 12, 128]
            for r in range(1, 4):
                conv3_x = conv_res_block(conv3_x, FILTERS[2], KERNEL_SIZE, 1, 'x'+str(frame)+'_3b'+str(r))
            print("BLOCK2 - X" + str(frame)+": ", conv3_x.shape)
            conv_stack3.append(conv3_x)

        # residual block3
        conv_stack4 = []
        for m in range(0, inp_cnt):
            frame = m+1
            conv4_x = conv_res_block(conv_stack3[m], FILTERS[3], KERNEL_SIZE, 1, 'x'+str(frame)+'_4a', identity_connection=False) # [batch_size, 56, 12, 256]
            for r1 in range(1, 23):
                conv4_x = conv_res_block(conv4_x, FILTERS[3], KERNEL_SIZE, 1, 'x'+str(frame)+'_4b'+str(r1))
            print("BLOCK3 - X" + str(frame)+": ", conv4_x.shape)
            conv_stack4.append(conv4_x)

        # residual block4
        conv_stack5 = []
        for n in range(0, inp_cnt):
            frame = n+1
            conv5_x = conv_res_block(conv_stack4[n], FILTERS[4], KERNEL_SIZE, 2, 'x'+str(frame)+'_5a', identity_connection=False) # [batch_size, 28, 6, 512]
            conv5_x = conv_res_block(conv5_x, FILTERS[4], KERNEL_SIZE, 1, 'x'+str(frame)+'_5b')
            conv5_x = conv_res_block(conv5_x, FILTERS[4], KERNEL_SIZE, 1, 'x'+str(frame)+'_5c')
            print("BLOCK4 - X" + str(frame)+": ", conv5_x.shape)
            conv_stack5.append(conv5_x)

        #return convlstm1_s, convlstm2_s, convlstm3_s, convlstm4_s, conv_stack5
        return conv_stack1[-1], conv_stack2[-1], conv_stack3[-1], conv_stack5
'''
def encoder_conv_net(x, mode):
    with tf.device('/gpu:0'):
        #config = tf.ConfigProto(allow_soft_placement=True)
        training = tf.equal(mode, 'train')
        inp_cnt = len(x)
        # ResNet101 as a backbone structure for encoder
        # strating block
        # 1st conv + relu + bn + convlstm block
        conv_stack1 = []
        for i in range(0, inp_cnt):
            conv1_x = start_conv_block(x[i], FILTERS[1], KERNEL_SIZE, 2, str(i+1))  #[batch_size, 225, 50, 16]
            print('conv1_x: ', conv1_x.shape)
            conv_stack1.append(conv1_x)

        # shape for convolved inputs that flows through conv LSTMs
        feat_shape = conv1_x.get_shape().as_list()[1:]
        #print('feat_shape: ', feat_shape)
        convlstm1_o, convlstm1_s = convlstm_block(feat_shape, conv_stack1, KERNEL_SIZE, None, 'convlstm_skip1') #[batch_size, 225, 50, 16]
        print('convlstm1_s: ', convlstm1_s)

        pool1_stack = []
        for j in range(0, inp_cnt):
            pool1_x = start_pool_block(conv_stack1[j], (KERNEL_SIZE, 2), (2,2), str(j+1))  #[batch_size, 112, 25, 16]
            print('pool1_x: ', pool1_x.shape)
            pool1_stack.append(pool1_x)

        # residual block1
        conv_stack2 = []
        for k in range(0, inp_cnt):
            frame = k+1
            conv2_x = conv_res_block(pool1_stack[k], FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frame)+'_2a', identity_connection=False) # [batch_size, 112, 25, 64]
            conv2_x = conv_res_block(conv2_x, FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frame)+'_2b')
            conv2_x = conv_res_block(conv2_x, FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frame)+'_2c')
            print("BLOCK1 - X" + str(frame)+": ", conv2_x.shape)
            conv_stack2.append(conv2_x)

        # shape for convolved inputs that flows through conv LSTMs
        feat_shape = conv2_x.get_shape().as_list()[1:]
        #print('feat_shape: ', feat_shape)
        convlstm2_o, convlstm2_s = convlstm_block(feat_shape, conv_stack2, KERNEL_SIZE, None, 'convlstm_skip2') #[batch_size, 112, 25, 64]
        print('convlstm2_s: ', convlstm2_s)

        # residual block2
        conv_stack3 = []
        for l in range(0, inp_cnt):
            frame = l+1
            conv3_x = conv_res_block(conv_stack2[l], FILTERS[2], KERNEL_SIZE, 2, 'x'+str(frame)+'_3a', identity_connection=False) # [batch_size, 56, 12, 128]
            for r in range(1, 4):
                conv3_x = conv_res_block(conv3_x, FILTERS[2], KERNEL_SIZE, 1, 'x'+str(frame)+'_3b'+str(r))
            print("BLOCK2 - X" + str(frame)+": ", conv3_x.shape)
            conv_stack3.append(conv3_x)

        # shape for convolved inputs that flows through conv LSTMs
        feat_shape = conv3_x.get_shape().as_list()[1:]
        #print('feat_shape: ', feat_shape)
        convlstm3_o, convlstm3_s = convlstm_block(feat_shape, conv_stack3, KERNEL_SIZE, None, 'convlstm_skip3') # [batch_size, 56, 12, 128]
        print('convlstm3_s: ', convlstm3_s)

        # residual block3
        conv_stack4 = []
        for m in range(0, inp_cnt):
            frame = m+1
            conv4_x = conv_res_block(conv_stack3[m], FILTERS[3], KERNEL_SIZE, 1, 'x'+str(frame)+'_4a', identity_connection=False) # [batch_size, 56, 12, 256]
            for r1 in range(1, 23):
                conv4_x = conv_res_block(conv4_x, FILTERS[3], KERNEL_SIZE, 1, 'x'+str(frame)+'_4b'+str(r1))
            print("BLOCK3 - X" + str(frame)+": ", conv4_x.shape)
            conv_stack4.append(conv4_x)

        # residual block4
        conv_stack5 = []
        for n in range(0, inp_cnt):
            frame = n+1
            conv5_x = conv_res_block(conv_stack4[n], FILTERS[4], KERNEL_SIZE, 2, 'x'+str(frame)+'_5a', identity_connection=False) # [batch_size, 28, 6, 512]
            conv5_x = conv_res_block(conv5_x, FILTERS[4], KERNEL_SIZE, 1, 'x'+str(frame)+'_5b')
            conv5_x = conv_res_block(conv5_x, FILTERS[4], KERNEL_SIZE, 1, 'x'+str(frame)+'_5c')
            print("BLOCK4 - X" + str(frame)+": ", conv5_x.shape)
            conv_stack5.append(conv5_x)

        #return convlstm1_s, convlstm2_s, convlstm3_s, convlstm4_s, conv_stack5
        return convlstm1_o, convlstm2_o, convlstm3_o, conv_stack5

def get_lstm_stack(lstm_output, shape):
    output1 = tf.split(lstm_output, tf.ones((NO_OF_FRAMES), dtype=tf.int32 ), 1)
    #print('output1: ', output1)
    output_stack = []
    for i in range(NO_OF_FRAMES):
        out= output1[i]
        out = tf.reshape(out, [BATCH_SIZE, shape[0], shape[1], shape[2]])
        output_stack.append(out)
    return output_stack

def get_lstm_stack1(lstm_output, shape, frame_no):
    output1 = tf.split(lstm_output, tf.ones((frame_no), dtype=tf.int32 ), 1)
    #print('output1: ', output1)
    output_stack = []
    for i in range(frame_no):
        out= output1[i]
        out = tf.reshape(out, [BATCH_SIZE, shape[0], shape[1], shape[2]])
        output_stack.append(out)
    return output_stack

def encoder_lstm(conv_stack, mode):
    training = tf.equal(mode, 'train')
    #print('conv_stack: ', conv_stack)
    feat_shape = conv_stack[-1].get_shape().as_list()[1:]
    #print('feat_shape: ', feat_shape)
    convlstm_output1, convlstm_state1 = convlstm_block(feat_shape, conv_stack, KERNEL_SIZE, None, 'convlstm_1') # [batch_size, 28, 6, 256]
    print('convlstm_state1: ', convlstm_state1)
    return convlstm_output1, convlstm_state1

def decoder_lstm(y, h1, seq, mode):
    #print(y)
    training = tf.equal(mode, 'train')
    feat_shape = y.get_shape().as_list()[1:]
    output1, state1  = dec_convlstm_block(feat_shape, y, KERNEL_SIZE, h1, 'decoder_convlstm_1'+str(seq)) # [batch_size, 28, 6, 256]
    return output1, state1

def recurrent_skip(y, seq_name):
    feat_shape = y.get_shape().as_list()[1:]
    output1, state1  = skip_convlstm_block(feat_shape, y, KERNEL_SIZE, None, seq_name)
    #print('skip_convlstm_output: ', state1)
    return output1, state1

def upsampling_network(lstm_output_cur, feat_shape, skip1, skip2, skip3, seq, mode):
    with tf.device('/gpu:0'):
        conv_feat = conv2d_layer(lstm_output_cur, RES_FILTERS[4], 1, 1, PADDING_SAME) # [batch_size, 28, 6, 512]
        print('conv_feat: ', conv_feat.shape)

        up1 = tf.layers.conv2d_transpose(inputs=conv_feat, filters=RES_FILTERS[2], kernel_size=[3, 3], padding="same", strides=2, activation=tf.nn.relu) # [batch_size, 56, 12, 128]
        print('up1: ', up1.shape)
        up1 = tf.add(up1, skip3, name="skip_1")
        up1 = conv2d_layer(up1, RES_FILTERS[2], 1, 1, PADDING_SAME)
        rskip3, _ = recurrent_skip(up1, 'skip_convlstm1_'+str(seq))

        up2 = tf.layers.conv2d_transpose(inputs=up1, filters=RES_FILTERS[1], kernel_size=[2, 3], padding="valid", strides=2, activation=tf.nn.relu) # [batch_size, 112, 25, 64]
        print('up2: ', up2.shape)
        up2 = tf.add(up2, skip2, name="skip_2")
        up2 = conv2d_layer(up2, RES_FILTERS[1], 1, 1, PADDING_SAME)
        rskip2, _ = recurrent_skip(up2, 'skip_convlstm2_'+str(seq))

        up3 = tf.layers.conv2d_transpose(inputs=up2, filters=FILTERS[1], kernel_size=[3, 2], padding="valid", strides=2, activation=tf.nn.relu) # [batch_size, 225, 50, 16]
        print('up3: ', up3.shape)
        up3 = tf.add(up3, skip1, name="skip_3")
        up3 = conv2d_layer(up3, FILTERS[1], 1, 1, PADDING_SAME)
        rskip1, _ = recurrent_skip(up3, 'skip_convlstm3_'+str(seq))

        up4 = tf.layers.conv2d_transpose(inputs=up3, filters=FILTERS[1], kernel_size=[3, 3], padding="same", strides=2, activation=tf.nn.relu) # [batch_size, 450, 100, 16]
        print('up4: ', up4.shape)

        logits = tf.layers.conv2d(inputs=up4, filters=1, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=(1, 1), padding='same', activation=tf.nn.relu) # [batch_size, 450, 100, 1]
        print('logits: ', logits.shape)

        pred = tf.layers.conv2d(inputs=logits, filters=1, kernel_size=[1, 1], strides=(1, 1), padding='same', activation=tf.nn.sigmoid) # [batch_size, 450, 100, 1]
        print('predictions: ', pred.shape)
        return logits, pred, rskip1, rskip2, rskip3

def read_samples():
    TRAIN_SET = []
    VALIDATION_SET = []
    TEST_SET = []
    TRAIN_VAL_SET = []

    file_list = glob.glob(DATASET_DIR + "*.npz")
    sorted_list = sorted(file_list)
    #print('sorted_list {}'.format(sorted_list))

    # divide dataset into 2 sets: i) train+validation set ii) test set
    # and shuffle only set (i) and later divide shuffle set into train+validation sets.
    for sample in range(0, TRAIN_SAMPLES + VALIDATION_SAMPLES):
        TRAIN_VAL_SET.append(sorted_list[sample])
    #random.shuffle(TRAIN_VAL_SET)

    for i in range(0, TRAIN_SAMPLES):
        TRAIN_SET.append(sorted_list[i])
        #TRAIN_SET.append(TRAIN_VAL_SET[i])
    #print('Sorted TRAIN LIST: ', TRAIN_SET)

    for j in range(TRAIN_SAMPLES, TRAIN_SAMPLES + VALIDATION_SAMPLES):
        VALIDATION_SET.append(sorted_list[j])
        #VALIDATION_SET.append(TRAIN_VAL_SET[j])

    for k in range(TRAIN_SAMPLES + VALIDATION_SAMPLES, TRAIN_SAMPLES + VALIDATION_SAMPLES + TEST_SAMPLES):
        TEST_SET.append(sorted_list[k])

    random.shuffle(TRAIN_SET)
    #print('shuffle_train_list: {}'.format(TRAIN_SET))
    return TRAIN_SET, VALIDATION_SET, TEST_SET

def read_sample_data(path):
    with np.load(path) as data:
        #print('sample - ', path)
        return data['arr_0']

def dice_coeff(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, dtype=tf.bool)
    y_pred = tf.cast(y_pred, dtype=tf.bool)
    intersection = tf.cast(tf.logical_and(y_true, y_pred), dtype=tf.int32)
    union = tf.cast(tf.logical_or(y_true, y_pred), dtype=tf.int32)
    score = tf.cast(tf.reduce_sum(2 * intersection)/tf.reduce_sum(union) + smooth, dtype=tf.float32)

    #intersection = tf.cast(tf.reduce_sum(y_true * y_pred), dtype=tf.float32)
    #score = ((2.0 * intersection) + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return score

def original_dice_coef(tp, fp, fn):
    smooth = 1e-6
    return (2 * tf.reduce_sum(tp) / (tf.reduce_sum(2*tp) + tf.reduce_sum(fp) + tf.reduce_sum(fn) + smooth))

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    loss = 1 - score
    return loss

def original_dice_loss(tp, tn, fp, fn):
    smooth = 1
    return (2 * tf.reduce_sum(tp) / (tf.reduce_sum(2*tp) + tf.reduce_sum(fp) + tf.reduce_sum(fn) + smooth))

def weighted_bce_loss(y_true, y_pred):
    loss = BETA * (y_true * (-tf.log(y_pred + eps))) + (1 - BETA) * ((1 - y_true) * (-tf.log((1 - y_pred) + eps)))
    return loss

def bce_loss(y_true, y_pred):
    loss = (y_true * (-tf.log(y_pred + eps))) + ((1 - y_true) * (-tf.log((1 - y_pred) + eps)))
    return loss

def get_confusion_matrix(y_pred, y_true):
    ones_like_actuals = tf.ones_like(y_true)
    zeros_like_actuals = tf.zeros_like(y_true)
    ones_like_predictions = tf.ones_like(y_pred)
    zeros_like_predictions = tf.zeros_like(y_pred)

    TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, ones_like_actuals), tf.equal(y_pred, ones_like_predictions)), dtype=tf.float32))
    TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, zeros_like_actuals), tf.equal(y_pred, zeros_like_predictions)), dtype=tf.float32))
    FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, zeros_like_actuals), tf.equal(y_pred, ones_like_predictions)), dtype=tf.float32))
    FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, ones_like_actuals), tf.equal(y_pred, zeros_like_predictions)), dtype=tf.float32))

    #TP = tf.cast(tf.count_nonzero(pred_int * labels, dtype=tf.int32), dtype=tf.float32)
    #TN = tf.cast(tf.count_nonzero((pred_int - 1) * (labels - 1), dtype=tf.int32), dtype=tf.float32)
    #FP = tf.cast(tf.count_nonzero(pred_int * (labels - 1), dtype=tf.int32), dtype=tf.float32)
    #FN = tf.cast(tf.count_nonzero((pred_int - 1) * labels, dtype=tf.int32), dtype=tf.float32)

    return TP, TN, FP, FN

def get_accuracy_recall_precision_f1score(tp, tn, fp, fn):
    # accuracy ::= (TP + TN) / (TN + FN + TP + FP)
    accuracy = tf.cast(tf.divide(tp + tn, tp + tn + fp + fn, name="Accuracy"), dtype=tf.float32)
    # precision ::= TP / (TP + FP)
    precision = tf.cast(tf.divide(tp, tp + fp, name="Precision"), dtype=tf.float32)
    precision = tf.where(tf.is_nan(precision), 0., precision)
    # recall ::= TP / (TP + FN)
    recall = tf.cast(tf.divide(tp, tp + fn, name="Recall"), dtype=tf.float32)
    recall = tf.where(tf.is_nan(recall), 0., recall)
    # F1 score ::= 2 * precision * recall / (precision + recall)
    f1 = tf.cast(tf.divide((2 * precision * recall), (precision + recall), name="F1_score"), dtype=tf.float32)

    TPR = 1. * tp / (tp + fn)
    FPR = 1. * fp / (fp + tn)
    return accuracy, precision, recall, f1

def tversky_loss(lab, pred, alpha=0.5, beta=0.5):
    EPSILON = 0.00001
    TP = tf.cast(tf.reduce_sum(lab * pred), dtype=tf.float32)
    TN = tf.cast(tf.reduce_sum((pred - 1) * (lab - 1)), dtype=tf.float32)
    FP = tf.cast(tf.reduce_sum(pred * (lab - 1)), dtype=tf.float32)
    FN = tf.cast(tf.reduce_sum((pred - 1) * lab), dtype=tf.float32)

    fp = alpha * FP
    fn = beta * FN
    numerator = TP
    denominator = TP + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)

def mean_iou_metric(true_y, pred_y):
    EPSILON = 0.00001
    intersection = tf.cast(tf.reduce_sum(true_y * pred_y), dtype=tf.float32)
    union = tf.cast((tf.reduce_sum(true_y) + tf.reduce_sum(pred_y)), dtype=tf.float32)
    iou = (intersection + EPSILON) / (union + EPSILON)
    mean_iou = tf.reduce_mean(iou)
    return mean_iou

def compute_threshold(frame, th):
    threshold = th
    frame[frame > th] = 1
    frame[frame <= th] = 0
    return frame

def main(unused_argv):
    TRAIN_FEATURES = []
    TRAIN_LABELS = []
    VALIDATION_FEATURES = []
    VALIDATION_LABELS = []
    TEST_FEATURES = []
    TEST_LABELS = []
    gpu_fraction = 0.95
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth = True), allow_soft_placement=True)
    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.device('/gpu:0'):
        # tf Graph input
        x1_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        x2_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        x3_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        x4_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        x5_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])

        y1_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        y2_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        y3_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        y4_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        y5_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        lr_placeholder = tf.placeholder(tf.float32)
        mode = tf.placeholder(tf.string, name='mode')
        logits = []
        preds = []

        input_stack = [x1_placeholder, x2_placeholder, x3_placeholder, x4_placeholder, x5_placeholder]
        print('ENCODER NETWORK...')
        #convlstm1_o, convlstm2_o, convlstm3_o, conv_last_stack = encoder_conv_net(input_stack, mode)
        skip1, skip2, skip3, conv_last_stack = encoder_conv_net(input_stack, mode)
        print('ENCODER LSTM...')
        output1, state1 = encoder_lstm(conv_last_stack, mode)

        '''
        # for t+1 frame prediction
        feat_shape = state1.h.get_shape().as_list()[1:]
        logit, pred, skip1, skip2, skip3 = upsampling_network(output1, feat_shape, skip1, skip2, skip3, 1, mode)
        logits.append(logit)
        preds.append(pred)
        '''

        dec_input = conv_last_stack[-1]
        print('DECODER NETWORK... Xn+1 time frame')
        dec_convlstm_o, dec_convlstm_s = decoder_lstm(dec_input, state1, 1, mode)
        feat_shape = dec_convlstm_s.h.get_shape().as_list()[1:]
        logit, pred, skip1, skip2, skip3 = upsampling_network(dec_convlstm_o, feat_shape, skip1, skip2, skip3, 1, mode)
        logits.append(logit)
        preds.append(pred)

        for dec in range(1, NO_OF_FRAMES):
            print('DECODER NETWORK... Xn+' + str(dec+1) + ' time frame')
            dec_convlstm_o, dec_convlstm_s = decoder_lstm(dec_convlstm_o, dec_convlstm_s, dec+1, mode)
            feat_shape = dec_convlstm_s.h.get_shape().as_list()[1:]
            logit, pred, skip1, skip2, skip3 = upsampling_network(dec_convlstm_o, feat_shape, skip1, skip2, skip3, dec+1, mode)
            logits.append(logit)
            preds.append(pred)

        logits1 = tf.reshape(logits[0], [BATCH_SIZE, -1],  name="logits_tensor1")
        print('logits1: ', logits[0])
        logits2 = tf.reshape(logits[1], [BATCH_SIZE, -1],  name="logits_tensor2")
        print('logits2: ', logits[1])
        logits3 = tf.reshape(logits[2], [BATCH_SIZE, -1],  name="logits_tensor3")
        print('logits3: ', logits[2])
        logits4 = tf.reshape(logits[3], [BATCH_SIZE, -1],  name="logits_tensor4")
        print('logits4: ', logits[3])
        logits5 = tf.reshape(logits[4], [BATCH_SIZE, -1],  name="logits_tensor5")
        print('logits5: ', logits[4])

        pred1 = tf.reshape(preds[0], [BATCH_SIZE, -1], name="sigmoid_tensor1")
        pred2 = tf.reshape(preds[1], [BATCH_SIZE, -1], name="sigmoid_tensor2")
        pred3 = tf.reshape(preds[2], [BATCH_SIZE, -1], name="sigmoid_tensor3")
        pred4 = tf.reshape(preds[3], [BATCH_SIZE, -1], name="sigmoid_tensor4")
        pred5 = tf.reshape(preds[4], [BATCH_SIZE, -1], name="sigmoid_tensor5")
        print('pred5: ', pred5)

        if mode != 'test':
            label1 = tf.reshape(y1_placeholder, [BATCH_SIZE, 45000])
            label2 = tf.reshape(y2_placeholder, [BATCH_SIZE, 45000])
            label3 = tf.reshape(y3_placeholder, [BATCH_SIZE, 45000])
            label4 = tf.reshape(y4_placeholder, [BATCH_SIZE, 45000])
            label5 = tf.reshape(y5_placeholder, [BATCH_SIZE, 45000])

            # weighted bce + dice loss
            loss1 = ALPHA * weighted_bce_loss(label1, pred1) + (1 - ALPHA) * dice_loss(label1, pred1)
            loss2 = ALPHA * weighted_bce_loss(label2, pred2) + (1 - ALPHA) * dice_loss(label2, pred2)
            loss3 = ALPHA * weighted_bce_loss(label3, pred3) + (1 - ALPHA) * dice_loss(label3, pred3)
            loss4 = ALPHA * weighted_bce_loss(label4, pred4) + (1 - ALPHA) * dice_loss(label4, pred4)
            loss5 = ALPHA * weighted_bce_loss(label5, pred5) + (1 - ALPHA) * dice_loss(label5, pred5)

            loss1 = tf.reduce_mean(loss1, name='loss1')
            loss2 = tf.reduce_mean(loss2, name='loss2')
            loss3 = tf.reduce_mean(loss3, name='loss3')
            loss4 = tf.reduce_mean(loss4, name='loss4')
            loss5 = tf.reduce_mean(loss5, name='loss5')
            loss = [loss1, loss2, loss3, loss4, loss5]
            loss = tf.reduce_mean(loss)
            print('loss: ', loss)

            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_placeholder)
            #optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_placeholder)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            # training metrics
            ## Round our sigmoid [0,1] output to be either 0 or 1
            pred_integer1 = tf.cast(tf.round(pred1), tf.int32)
            pred_integer2 = tf.cast(tf.round(pred2), tf.int32)
            pred_integer3 = tf.cast(tf.round(pred3), tf.int32)
            pred_integer4 = tf.cast(tf.round(pred4), tf.int32)
            pred_integer5 = tf.cast(tf.round(pred5), tf.int32)

            labels_integer1 = tf.cast(label1, tf.int32)
            labels_integer2 = tf.cast(label2, tf.int32)
            labels_integer3 = tf.cast(label3, tf.int32)
            labels_integer4 = tf.cast(label4, tf.int32)
            labels_integer5 = tf.cast(label5, tf.int32)

            ## Define operations for {true, false} {positive, negative} predictions
            TP1, TN1, FP1, FN1 = get_confusion_matrix(pred_integer1, labels_integer1)
            TP2, TN2, FP2, FN2 = get_confusion_matrix(pred_integer2, labels_integer2)
            TP3, TN3, FP3, FN3 = get_confusion_matrix(pred_integer3, labels_integer3)
            TP4, TN4, FP4, FN4 = get_confusion_matrix(pred_integer4, labels_integer4)
            TP5, TN5, FP5, FN5 = get_confusion_matrix(pred_integer5, labels_integer5)

            print('TP, TN, FP, FN: ', TP1, TN1, FP1, FN1)
            accuracy_op1, precision_op1, recall_op1, f11 = get_accuracy_recall_precision_f1score(TP1, TN1, FP1, FN1)
            accuracy_op2, precision_op2, recall_op2, f12 = get_accuracy_recall_precision_f1score(TP2, TN2, FP2, FN2)
            accuracy_op3, precision_op3, recall_op3, f13 = get_accuracy_recall_precision_f1score(TP3, TN3, FP3, FN3)
            accuracy_op4, precision_op4, recall_op4, f14 = get_accuracy_recall_precision_f1score(TP4, TN4, FP4, FN4)
            accuracy_op5, precision_op5, recall_op5, f15 = get_accuracy_recall_precision_f1score(TP5, TN5, FP5, FN5)
            accuracy = [accuracy_op1, accuracy_op2, accuracy_op3, accuracy_op4, accuracy_op5]
            accuracy = tf.reduce_mean(accuracy)

            precision = [precision_op1, precision_op2, precision_op3, precision_op4, precision_op5]
            precision = tf.reduce_mean(precision)
            recall = [recall_op1, recall_op2, recall_op3, recall_op4, recall_op5]
            recall = tf.reduce_mean(recall)
            print('accuracy_op, precision_op, recall_op, f1: ', accuracy_op1, precision_op1, recall_op1, f11)

            dice_coef1 = dice_coeff(labels_integer1, pred_integer1)
            dice_coef2 = dice_coeff(labels_integer2, pred_integer2)
            dice_coef3 = dice_coeff(labels_integer3, pred_integer3)
            dice_coef4 = dice_coeff(labels_integer4, pred_integer4)
            dice_coef5 = dice_coeff(labels_integer5, pred_integer5)
            dice_coef = [dice_coef1, dice_coef2, dice_coef3, dice_coef4, dice_coef5]
            dice_coef = tf.reduce_mean(dice_coef)

            is_correct1 = tf.equal(pred_integer1, labels_integer1)
            is_correct2 = tf.equal(pred_integer2, labels_integer2)
            is_correct3 = tf.equal(pred_integer3, labels_integer3)
            is_correct4 = tf.equal(pred_integer4, labels_integer4)
            is_correct5 = tf.equal(pred_integer5, labels_integer5)

            pixel_accuracy1 = tf.reduce_mean(tf.cast(is_correct1, tf.float32), name='pixel_accuracy1')
            pixel_accuracy2 = tf.reduce_mean(tf.cast(is_correct2, tf.float32), name='pixel_accuracy2')
            pixel_accuracy3 = tf.reduce_mean(tf.cast(is_correct3, tf.float32), name='pixel_accuracy3')
            pixel_accuracy4 = tf.reduce_mean(tf.cast(is_correct4, tf.float32), name='pixel_accuracy4')
            pixel_accuracy5 = tf.reduce_mean(tf.cast(is_correct5, tf.float32), name='pixel_accuracy5')
            pixel_accuracy = [pixel_accuracy1, pixel_accuracy2, pixel_accuracy3, pixel_accuracy4, pixel_accuracy5]
            pixel_accuracy = tf.reduce_mean(pixel_accuracy)

            mean_class_accuracy1, mean_class_acc_op1 = tf.metrics.mean_per_class_accuracy(labels_integer1, pred_integer1, 2, name='mean_class_accuracy1')
            mean_class_accuracy2, mean_class_acc_op2 = tf.metrics.mean_per_class_accuracy(labels_integer2, pred_integer2, 2, name='mean_class_accuracy2')
            mean_class_accuracy3, mean_class_acc_op3 = tf.metrics.mean_per_class_accuracy(labels_integer3, pred_integer3, 2, name='mean_class_accuracy3')
            mean_class_accuracy4, mean_class_acc_op4 = tf.metrics.mean_per_class_accuracy(labels_integer4, pred_integer4, 2, name='mean_class_accuracy4')
            mean_class_accuracy5, mean_class_acc_op5 = tf.metrics.mean_per_class_accuracy(labels_integer5, pred_integer5, 2, name='mean_class_accuracy5')
            mean_class_accuracy = [mean_class_accuracy1, mean_class_accuracy2, mean_class_accuracy3, mean_class_accuracy4, mean_class_accuracy5]
            mean_class_accuracy = tf.reduce_mean(mean_class_accuracy)

            mean_iou1, mean_op1 = tf.metrics.mean_iou(labels_integer1, pred_integer1, 2, name='mean_iou1')
            mean_iou2, mean_op2 = tf.metrics.mean_iou(labels_integer2, pred_integer2, 2, name='mean_iou2')
            mean_iou3, mean_op3 = tf.metrics.mean_iou(labels_integer3, pred_integer3, 2, name='mean_iou3')
            mean_iou4, mean_op4 = tf.metrics.mean_iou(labels_integer4, pred_integer4, 2, name='mean_iou4')
            mean_iou5, mean_op5 = tf.metrics.mean_iou(labels_integer5, pred_integer5, 2, name='mean_iou5')
            mean_iou = [mean_iou1, mean_iou2, mean_iou3, mean_iou4, mean_iou5]
            mean_iou = tf.reduce_mean(mean_iou)

            auc1, auc_op1 = tf.metrics.auc(labels_integer1, pred_integer1, name='auc1')
            auc2, auc_op2 = tf.metrics.auc(labels_integer2, pred_integer2, name='auc2')
            auc3, auc_op3 = tf.metrics.auc(labels_integer3, pred_integer3, name='auc3')
            auc4, auc_op4 = tf.metrics.auc(labels_integer4, pred_integer4, name='auc4')
            auc5, auc_op5 = tf.metrics.auc(labels_integer5, pred_integer5, name='auc5')
            auc = [auc1, auc2, auc3, auc4, auc5]
            auc = tf.reduce_mean(auc)

            #mean_iou1, mean_op1 = mean_iou_metric(labels_integer1, pred_integer1)
            #mean_iou2, mean_op2 = mean_iou_metric(labels_integer2, pred_integer2)
            #mean_iou3, mean_op3 = mean_iou_metric(labels_integer3, pred_integer3)

            with tf.name_scope('net_model_summary'):
                summary_loss = tf.summary.scalar('loss', loss)
                summary_pre = tf.summary.scalar('precision', precision)
                summary_rec = tf.summary.scalar('recall', recall)
                summary_miou = tf.summary.scalar('mean_iou', mean_iou)
                summary_dice_coef = tf.summary.scalar('dice_coef', dice_coef)
                summary_class_acc = tf.summary.scalar('mean_class_accuracy', mean_class_accuracy)
                summary_acc = tf.summary.scalar('accuracy', accuracy)
                summary_auc = tf.summary.scalar('auc', auc)

                summary_pre1 = tf.summary.scalar('precision1', precision_op1)
                summary_pre2 = tf.summary.scalar('precision2', precision_op2)
                summary_pre3 = tf.summary.scalar('precision3', precision_op3)
                summary_pre4 = tf.summary.scalar('precision4', precision_op4)
                summary_pre5 = tf.summary.scalar('precision5', precision_op5)

                summary_rec1 = tf.summary.scalar('recall1', recall_op1)
                summary_rec2 = tf.summary.scalar('recall2', recall_op2)
                summary_rec3 = tf.summary.scalar('recall3', recall_op3)
                summary_rec4 = tf.summary.scalar('recall4', recall_op4)
                summary_rec5 = tf.summary.scalar('recall5', recall_op5)

                summary_miou1 = tf.summary.scalar('mean_iou1', mean_iou1)
                summary_miou2 = tf.summary.scalar('mean_iou2', mean_iou2)
                summary_miou3 = tf.summary.scalar('mean_iou3', mean_iou3)
                summary_miou4 = tf.summary.scalar('mean_iou3', mean_iou4)
                summary_miou5 = tf.summary.scalar('mean_iou3', mean_iou5)

                summary_auc1 = tf.summary.scalar('auc1', auc1)
                summary_auc2 = tf.summary.scalar('auc2', auc2)
                summary_auc3 = tf.summary.scalar('auc3', auc3)
                summary_auc4 = tf.summary.scalar('auc4', auc4)
                summary_auc5 = tf.summary.scalar('auc5', auc5)

                summary_dice_coef1 = tf.summary.scalar('dice_coef1', dice_coef1)
                summary_dice_coef2 = tf.summary.scalar('dice_coef2', dice_coef2)
                summary_dice_coef3 = tf.summary.scalar('dice_coef3', dice_coef3)
                summary_dice_coef4 = tf.summary.scalar('dice_coef4', dice_coef4)
                summary_dice_coef5 = tf.summary.scalar('dice_coef5', dice_coef5)

                summary_class_acc1 = tf.summary.scalar('mean_class_accuracy1', mean_class_accuracy1)
                summary_class_acc2 = tf.summary.scalar('mean_class_accuracy2', mean_class_accuracy2)
                summary_class_acc3 = tf.summary.scalar('mean_class_accuracy3', mean_class_accuracy3)
                summary_class_acc4 = tf.summary.scalar('mean_class_accuracy4', mean_class_accuracy4)
                summary_class_acc5 = tf.summary.scalar('mean_class_accuracy5', mean_class_accuracy5)

                summary_pixacc1 = tf.summary.scalar('pixel_accuracy1', pixel_accuracy1)
                summary_pixacc2 = tf.summary.scalar('pixel_accuracy2', pixel_accuracy2)
                summary_pixacc3 = tf.summary.scalar('pixel_accuracy3', pixel_accuracy3)
                summary_pixacc4 = tf.summary.scalar('pixel_accuracy4', pixel_accuracy4)
                summary_pixacc5 = tf.summary.scalar('pixel_accuracy5', pixel_accuracy5)
                summary_pixacc = tf.summary.scalar('pixel_accuracy', pixel_accuracy)

                summary_acc1 = tf.summary.scalar('accuracy1', accuracy_op1)
                summary_acc2 = tf.summary.scalar('accuracy2', accuracy_op2)
                summary_acc3 = tf.summary.scalar('accuracy3', accuracy_op3)
                summary_acc4 = tf.summary.scalar('accuracy4', accuracy_op4)
                summary_acc5 = tf.summary.scalar('accuracy5', accuracy_op5)

                #summary_op = tf.summary.merge([summary_loss, summary_pixacc, summary_class_acc, summary_miou, summary_acc, summary_pre, summary_rec], name='network_summary')
                summary_op = tf.summary.merge([summary_loss, summary_pre, summary_rec, summary_miou, summary_dice_coef, summary_acc, summary_auc,
                                               summary_pre1, summary_pre2, summary_pre3, summary_pre4, summary_pre5,
                                               summary_rec1, summary_rec2, summary_rec3, summary_rec4, summary_rec5,
                                               summary_miou1, summary_miou2, summary_miou3, summary_miou4, summary_miou5,
                                               summary_auc1, summary_auc2, summary_auc3, summary_auc4, summary_auc5,
                                               summary_dice_coef1, summary_dice_coef2, summary_dice_coef3, summary_dice_coef4, summary_dice_coef5,
                                               summary_acc1, summary_acc2, summary_acc3, summary_acc4, summary_acc5], name='net_model_summary')

        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        saver = tf.train.Saver(max_to_keep=2)

        with tf.Session(config=config) as sess:
            sess.run(init)
            sess.run(init_local)
            train_writer = tf.summary.FileWriter(MODEL_DIR + '/train', sess.graph)
            validation_writer = tf.summary.FileWriter(MODEL_DIR + '/validation')
            test_writer = tf.summary.FileWriter(MODEL_DIR + '/test')

            best_val_loss = np.inf
            best_train_loss = np.inf
            best_miou = np.inf

            non_update_counter = 0
            train_model_update_counter = 0
            for epoch in range(EPOCHS):
                start = time.time()
                ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
                print('ckpt:', ckpt)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("Model restored...")
                total_loss_value = 0
                total_precision_value1 = 0
                total_precision_value2 = 0
                total_precision_value3 = 0
                total_precision_value4 = 0
                total_precision_value5 = 0
                total_precision_value = 0

                total_recall_value1 = 0
                total_recall_value2 = 0
                total_recall_value3 = 0
                total_recall_value4 = 0
                total_recall_value5 = 0
                total_recall_value = 0

                total_miou_value1 = 0
                total_miou_value2 = 0
                total_miou_value3 = 0
                total_miou_value4 = 0
                total_miou_value5 = 0
                total_miou_value = 0

                total_auc1 = 0
                total_auc2 = 0
                total_auc3 = 0
                total_auc4 = 0
                total_auc5 = 0
                total_auc = 0

                total_dice_value1 = 0
                total_dice_value2 = 0
                total_dice_value3 = 0
                total_dice_value4 = 0
                total_dice_value5 = 0
                total_dice_value = 0

                total_mean_class_acc_value = 0
                total_accuracy_value1 = 0
                total_accuracy_value2 = 0
                total_accuracy_value3 = 0
                total_accuracy_value4 = 0
                total_accuracy_value5 = 0
                total_accuracy_value = 0

                total_pix_acc_value1 = 0
                total_pix_acc_value2 = 0
                total_pix_acc_value3 = 0
                total_pix_acc_value4 = 0
                total_pix_acc_value5 = 0
                total_pix_acc_value = 0

                TRAIN_SET, VALIDATION_SET, TEST_SET = read_samples()
                iteration_no = int(len(TRAIN_SET)/BATCH_SIZE)
                print("Epoch: {}/{} - ".format(epoch+1, int(EPOCHS)))
                for batchno in range(0, int(len(TRAIN_SET)/BATCH_SIZE)):
                    t_set = []
                    for index in range(0, BATCH_SIZE):
                        t_set.append(read_sample_data(TRAIN_SET[(batchno * BATCH_SIZE) + index]))
                    for x in range(len(t_set)):
                        data_row = t_set[x][0]
                        TRAIN_FEATURES.append([data_row[0].todense(), data_row[1].todense(), data_row[2].todense(), data_row[3].todense(), data_row[4].todense()])
                        TRAIN_LABELS.append([data_row[5].todense(), data_row[6].todense(), data_row[7].todense(), data_row[8].todense(), data_row[9].todense()])
                    train_data = np.asarray(TRAIN_FEATURES, dtype=np.float32)
                    train_labels = np.asarray(TRAIN_LABELS, dtype=np.float32)
                    x1, x2, x3, x4, x5 = reshape_input(train_data)
                    y1, y2, y3, y4, y5 = reshape_input(train_labels)
                    feed = {x1_placeholder: x1, x2_placeholder: x2, x3_placeholder: x3, x4_placeholder: x4, x5_placeholder: x5,
                            y1_placeholder: y1, y2_placeholder: y2, y3_placeholder: y3, y4_placeholder: y4, y5_placeholder: y5, lr_placeholder: LEARNING_RATE, mode:'train'}

                    _, loss_value, pix_acc1, pix_acc2, pix_acc3, pix_acc4, pix_acc5, class_acc1, miou1, miou2, miou3, miou4, miou5, miou_, acc1, acc2, acc3, acc4, acc5, acc_, pre1, pre2, pre3, pre4, pre5, pre_, rec1, rec2, rec3, rec4, rec5, rec_, d1, d2, d3, d4, d5, dice, auc_roc1, auc_roc2, auc_roc3, auc_roc4, auc_roc5, auc_roc, summary_value = sess.run([train_op, loss,
                    pixel_accuracy1, pixel_accuracy2, pixel_accuracy3, pixel_accuracy4, pixel_accuracy5, mean_class_accuracy1, mean_iou1, mean_iou2, mean_iou3,
                    mean_iou4, mean_iou5, mean_iou, accuracy_op1, accuracy_op2, accuracy_op3, accuracy_op4, accuracy_op5, accuracy, precision_op1, precision_op2,
                    precision_op3, precision_op4, precision_op5, precision, recall_op1, recall_op2, recall_op3, recall_op4, recall_op5, recall, dice_coef1, dice_coef2, dice_coef3, dice_coef4, dice_coef5, dice_coef, auc1, auc2, auc3, auc4, auc5, auc, summary_op], feed_dict = feed)

                    *_, mop1, mop2, mop3, mop4, mop5, aucop1, aucop2, aucop3, aucop4, aucop5, tp1, tp2, tp3, tp4, tp5, fp1, fp2, fp3, fp4, fp5, fn1, fn2, fn3, fn4, fn5 = sess.run([mean_class_acc_op1, mean_op1, mean_op2, mean_op3, mean_op4, mean_op5, auc_op1, auc_op2, auc_op3, auc_op4, auc_op5, TP1, TP2, TP3, TP4, TP5, FP1, FP2, FP3, FP3, FP4, FP5, FN1, FN2, FN3, FN4, FN5], feed_dict = feed)

                    #sess.run([mean_class_acc_op1, mean_op1, mean_op2, mean_op3, mean_op4, mean_op5, accuracy_op1, accuracy_op2, accuracy_op3, accuracy_op4, accuracy_op5,
                              #precision_op1, precision_op2, precision_op3, precision_op4, precision_op5, recall_op1, recall_op2, recall_op3, recall_op4, recall_op5], feed_dict = feed)

                    total_loss_value += loss_value
                    total_precision_value1 += pre1
                    total_precision_value2 += pre2
                    total_precision_value3 += pre3
                    total_precision_value4 += pre4
                    total_precision_value5 += pre5
                    total_precision_value += pre_

                    total_recall_value1 += rec1
                    total_recall_value2 += rec2
                    total_recall_value3 += rec3
                    total_recall_value4 += rec4
                    total_recall_value5 += rec5
                    total_recall_value += rec_

                    total_miou_value1 += miou1
                    total_miou_value2 += miou2
                    total_miou_value3 += miou3
                    total_miou_value4 += miou4
                    total_miou_value5 += miou5
                    total_miou_value += miou_

                    total_auc1 += auc_roc1
                    total_auc2 += auc_roc2
                    total_auc3 += auc_roc3
                    total_auc4 += auc_roc4
                    total_auc5 += auc_roc5
                    total_auc += auc_roc

                    total_dice_value1 += d1
                    total_dice_value2 += d2
                    total_dice_value3 += d3
                    total_dice_value4 += d4
                    total_dice_value5 += d5
                    total_dice_value += dice

                    total_mean_class_acc_value += class_acc1
                    total_accuracy_value1 += acc1
                    total_accuracy_value2 += acc2
                    total_accuracy_value3 += acc3
                    total_accuracy_value4 += acc4
                    total_accuracy_value5 += acc5
                    total_accuracy_value += acc_

                    total_pix_acc_value1 += pix_acc1
                    total_pix_acc_value2 += pix_acc2
                    total_pix_acc_value3 += pix_acc3
                    total_pix_acc_value4 += pix_acc4
                    total_pix_acc_value5 += pix_acc5
                    train_writer.add_summary(summary_value, epoch * iteration_no + batchno)

                    #if (batchno + 1) % 100 == 0:
                        #print(sess.run([class_acc_op, mean_op, accuracy_op, precision_op, recall_op], feed_dict = feed))
                        #print("         Iteration: {}, Loss: {:.8f}, Accuracy: {:.8f}, Precision: {:.8f}, Recall: {:.8f}, PixelAccuracy: {:.8f}, MeanClassAccuracy: {:.8f}, Mean_IoU: {:.8f}".format(batchno+1,
                        #total_loss_value, total_accuracy_value, pre, rec, pix_acc, class_acc, miou))

                    '''#if (batchno + 1) % 100 == 0:
                        if total_loss_value < best_loss:
                            #best_loss = loss_value
                            save_path = saver.save(sess, MODEL_DIR + "model.ckpt")
                            #print("Model saved in file: %s" % save_path)
                            #print("    best model update!!!")'''

                    TRAIN_FEATURES = []
                    TRAIN_LABELS = []

                avg_loss = total_loss_value/iteration_no

                avg_precision1 = total_precision_value1/iteration_no
                avg_precision2 = total_precision_value2/iteration_no
                avg_precision3 = total_precision_value3/iteration_no
                avg_precision4 = total_precision_value4/iteration_no
                avg_precision5 = total_precision_value5/iteration_no
                #avg_precision = (avg_precision1 + avg_precision2 + avg_precision3 + avg_precision4 + avg_precision5)/5
                avg_precision = total_precision_value/iteration_no

                avg_recall1 = total_recall_value1/iteration_no
                avg_recall2 = total_recall_value2/iteration_no
                avg_recall3 = total_recall_value3/iteration_no
                avg_recall4 = total_recall_value4/iteration_no
                avg_recall5 = total_recall_value5/iteration_no
                #avg_recall = (avg_recall1 + avg_recall2 + avg_recall3 + avg_recall4 + avg_recall5)/5
                avg_recall = total_recall_value/iteration_no

                avg_miou1 = total_miou_value1/iteration_no
                avg_miou2 = total_miou_value2/iteration_no
                avg_miou3 = total_miou_value3/iteration_no
                avg_miou4 = total_miou_value4/iteration_no
                avg_miou5 = total_miou_value5/iteration_no
                #avg_miou = (avg_miou1 + avg_miou2 + avg_miou3 + avg_miou4 + avg_miou5)/5
                avg_miou = total_miou_value/iteration_no

                avg_auc1 = total_auc1/iteration_no
                avg_auc2 = total_auc2/iteration_no
                avg_auc3 = total_auc3/iteration_no
                avg_auc4 = total_auc4/iteration_no
                avg_auc5 = total_auc5/iteration_no
                avg_auc = total_auc/iteration_no

                avg_dice1 = total_dice_value1/iteration_no
                avg_dice2 = total_dice_value2/iteration_no
                avg_dice3 = total_dice_value3/iteration_no
                avg_dice4 = total_dice_value4/iteration_no
                avg_dice5 = total_dice_value5/iteration_no
                #avg_dice = (avg_dice1 + avg_dice2 + avg_dice3 + avg_dice4 + avg_dice5)/5
                avg_dice = total_dice_value/iteration_no

                avg_mean_class_acc = total_mean_class_acc_value/iteration_no
                avg_accuracy1 = total_accuracy_value1/iteration_no
                avg_accuracy2 = total_accuracy_value2/iteration_no
                avg_accuracy3 = total_accuracy_value3/iteration_no
                avg_accuracy4 = total_accuracy_value4/iteration_no
                avg_accuracy5 = total_accuracy_value5/iteration_no
                #avg_accuracy = (avg_accuracy1 + avg_accuracy2 + avg_accuracy3 + avg_accuracy4 + avg_accuracy5)/5
                avg_accuracy = total_accuracy_value/iteration_no

                avg_pix_acc1 = total_pix_acc_value1/iteration_no
                avg_pix_acc2 = total_pix_acc_value2/iteration_no
                avg_pix_acc3 = total_pix_acc_value3/iteration_no
                avg_pix_acc4 = total_pix_acc_value4/iteration_no
                avg_pix_acc5 = total_pix_acc_value5/iteration_no
                avg_pix_acc = (avg_pix_acc1 + avg_pix_acc2 + avg_pix_acc3 + avg_pix_acc4 + avg_pix_acc5)/5

                #end = time.time()
                #duration = end - start
                #print("Time elapsed: {0}".format(duration))
                #print("     Training Iteration: {}/{} | {:.2f}s - Loss: {:.8f}%, Accuracy: {:.8f}%, Precision: {:.8f}%, Recall: {:.8f}%, PixelAccuracy: {:.8f}%, MeanClassAccuracy: {:.3f}%, Mean_IoU: {:.3f}%".format(iteration_no,
                #iteration_no, duration, avg_loss * 100, avg_accuracy * 100, avg_precision * 100, avg_recall * 100, avg_pix_acc * 100, avg_mean_class_acc * 100, avg_miou * 100))

                if avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    #print("     training loss improved to {:.8f}% | model is saving...".format(best_train_loss * 100))
                    #save_path = saver.save(sess, MODEL_DIR + "model.ckpt")
                    train_model_update_counter = 0
                else:
                    train_model_update_counter += 1
                    #print("     training loss did not improve for the last {} epochs...".format(train_model_update_counter))


                #start = time.time()
                itr_validation = int(len(VALIDATION_SET)/BATCH_SIZE)
                #print("Validation => Epoch: {}/{} -  ".format(epoch+1, int(EPOCHS)))
                total_validation_loss_value = 0
                total_validation_pre_value1 = 0
                total_validation_pre_value2 = 0
                total_validation_pre_value3 = 0
                total_validation_pre_value4 = 0
                total_validation_pre_value5 = 0
                total_validation_pre_value = 0

                total_validation_rec_value1 = 0
                total_validation_rec_value2 = 0
                total_validation_rec_value3 = 0
                total_validation_rec_value4 = 0
                total_validation_rec_value5 = 0
                total_validation_rec_value = 0

                total_validation_miou_value1 = 0
                total_validation_miou_value2 = 0
                total_validation_miou_value3 = 0
                total_validation_miou_value4 = 0
                total_validation_miou_value5 = 0
                total_validation_miou_value = 0

                total_auc_v1 = 0
                total_auc_v2 = 0
                total_auc_v3 = 0
                total_auc_v4 = 0
                total_auc_v5 = 0
                total_auc_v = 0

                total_validation_dice_value1 = 0
                total_validation_dice_value2 = 0
                total_validation_dice_value3 = 0
                total_validation_dice_value4 = 0
                total_validation_dice_value5 = 0
                total_validation_dice_value = 0

                total_validation_acc_value1 = 0
                total_validation_acc_value2 = 0
                total_validation_acc_value3 = 0
                total_validation_acc_value4 = 0
                total_validation_acc_value5 = 0
                total_validation_acc_value = 0

                total_validation_pix_acc_value1 = 0
                total_validation_pix_acc_value2 = 0
                total_validation_pix_acc_value3 = 0
                total_validation_pix_acc_value4 = 0
                total_validation_pix_acc_value5 = 0
                total_validation_pix_acc_value = 0
                total_validation_mean_class_acc_value = 0

                for valno in range(0, int(len(VALIDATION_SET)/BATCH_SIZE)):
                    val_set = []
                    for idx in range(0, BATCH_SIZE):
                        val_set.append(read_sample_data(VALIDATION_SET[(valno * BATCH_SIZE) + idx]))
                    VALIDATION_FEATURES = []
                    VALIDATION_LABELS = []
                    for y in range(len(val_set)):
                        data_row = val_set[y][0]
                        VALIDATION_FEATURES.append([data_row[0].todense(), data_row[1].todense(), data_row[2].todense(), data_row[3].todense(), data_row[4].todense()])
                        VALIDATION_LABELS.append([data_row[5].todense(), data_row[6].todense(), data_row[7].todense(), data_row[8].todense(), data_row[9].todense()])

                    eval_data = np.asarray(VALIDATION_FEATURES, dtype=np.float32)
                    eval_labels = np.asarray(VALIDATION_LABELS, dtype=np.float32)
                    x1, x2, x3, x4, x5 = reshape_input(eval_data)
                    y1, y2, y3, y4, y5 = reshape_input(eval_labels)
                    feed = {x1_placeholder: x1, x2_placeholder: x2, x3_placeholder: x3, x4_placeholder: x4, x5_placeholder: x5,
                            y1_placeholder: y1, y2_placeholder: y2, y3_placeholder: y3, y4_placeholder: y4, y5_placeholder: y5, lr_placeholder: LEARNING_RATE,  mode:'validation'}

                    valid_loss, pix_acc1, pix_acc2, pix_acc3, pix_acc4, pix_acc5, class_acc1, miou1, miou2, miou3, miou4, miou5, miou_, valid_acc1, acc2, acc3, acc4, acc5, acc_, pre1, pre2, pre3, pre4, pre5, pre_, rec1, rec2, rec3, rec4, rec5, rec_, d1, d2, d3, d4, d5, dice, auc_roc1, auc_roc2, auc_roc3, auc_roc4, auc_roc5, auc_roc, summary_value_valid = sess.run([loss,
                    pixel_accuracy1, pixel_accuracy2, pixel_accuracy3, pixel_accuracy4, pixel_accuracy5, mean_class_accuracy1, mean_iou1, mean_iou2, mean_iou3, mean_iou4, mean_iou5, mean_iou, accuracy_op1, accuracy_op2, accuracy_op3,
                    accuracy_op4, accuracy_op5, accuracy, precision_op1, precision_op2, precision_op3, precision_op4, precision_op5, precision, recall_op1, recall_op2, recall_op3,
                    recall_op4, recall_op5, recall, dice_coef1, dice_coef2, dice_coef3, dice_coef4, dice_coef5, dice_coef, auc1, auc2, auc3, auc4, auc5, auc, summary_op], feed_dict= feed)

                    *_, mop1, mop2, mop3, mop4, mop5, aucop1, aucop2, aucop3, aucop4, aucop5, tp1, tp2, tp3, tp4, tp5, fp1, fp2, fp3, fp4, fp5, fn1, fn2, fn3, fn4, fn5 = sess.run([mean_class_acc_op1, mean_op1, mean_op2, mean_op3, mean_op4, mean_op5, auc_op1, auc_op2, auc_op3, auc_op4, auc_op5, TP1, TP2, TP3, TP4, TP5, FP1, FP2, FP3, FP3, FP4, FP5, FN1, FN2, FN3, FN4, FN5], feed_dict = feed)

                    '''
                    sess.run([mean_class_acc_op1, mean_op1, mean_op2, mean_op3, mean_op4, mean_op5, accuracy_op1, accuracy_op2, accuracy_op3, accuracy_op4, accuracy_op5, precision_op1, precision_op2, precision_op3,
                    precision_op4, precision_op5, recall_op1, recall_op2, recall_op3, recall_op4, recall_op5], feed_dict = feed)
                    '''

                    total_validation_loss_value += valid_loss
                    total_validation_pre_value1 += pre1
                    total_validation_pre_value2 += pre2
                    total_validation_pre_value3 += pre3
                    total_validation_pre_value4 += pre4
                    total_validation_pre_value5 += pre5
                    total_validation_pre_value += pre_

                    total_validation_rec_value1 += rec1
                    total_validation_rec_value2 += rec2
                    total_validation_rec_value3 += rec3
                    total_validation_rec_value4 += rec4
                    total_validation_rec_value5 += rec5
                    total_validation_rec_value += rec_

                    total_validation_miou_value1 += miou1
                    total_validation_miou_value2 += miou2
                    total_validation_miou_value3 += miou3
                    total_validation_miou_value4 += miou4
                    total_validation_miou_value5 += miou5
                    total_validation_miou_value += miou_

                    total_auc_v1 += auc_roc1
                    total_auc_v2 += auc_roc2
                    total_auc_v3 += auc_roc3
                    total_auc_v4 += auc_roc4
                    total_auc_v5 += auc_roc5
                    total_auc_v += auc_roc

                    total_validation_dice_value1 += d1
                    total_validation_dice_value2 += d2
                    total_validation_dice_value3 += d3
                    total_validation_dice_value4 += d4
                    total_validation_dice_value5 += d5
                    total_validation_dice_value += dice

                    total_validation_pix_acc_value1 += pix_acc1
                    total_validation_pix_acc_value2 += pix_acc2
                    total_validation_pix_acc_value3 += pix_acc3
                    total_validation_pix_acc_value4 += pix_acc4
                    total_validation_pix_acc_value5 += pix_acc5

                    total_validation_acc_value1 += valid_acc1
                    total_validation_acc_value2 += acc2
                    total_validation_acc_value3 += acc3
                    total_validation_acc_value4 += acc4
                    total_validation_acc_value5 += acc5
                    total_validation_acc_value += acc_

                    total_validation_mean_class_acc_value += class_acc1
                    validation_writer.add_summary(summary_value_valid, epoch * itr_validation + valno)

                    #if (valno+1) % 25 == 0:
                        #print("     Iteration: {}/{}, Loss: {:.8f}, Accuracy: {:.8f}, Precision: {:.8f}, Recall: {:.8f}, PixelAccuracy: {:.8f}, MeanClassAccuracy: {:.3f}, Mean_IoU: {:.3f}".format(valno+1,
                        #itr_validation, total_validation_loss_value, total_validation_acc_value, pre, rec, pix_acc, class_acc, miou))
                        #save_path = saver.save(sess, MODEL_DIR + "model.ckpt")
                        #saver.save(sess, MODEL_DIR + "model.ckpt")

                    VALIDATION_FEATURES = []
                    VALIDATION_LABELS = []

                avg_validation_loss = total_validation_loss_value/itr_validation
                avg_validation_pre1 = total_validation_pre_value1/itr_validation
                avg_validation_pre2 = total_validation_pre_value2/itr_validation
                avg_validation_pre3 = total_validation_pre_value3/itr_validation
                avg_validation_pre4 = total_validation_pre_value4/itr_validation
                avg_validation_pre5 = total_validation_pre_value5/itr_validation
                #avg_validation_pre = (avg_validation_pre1 + avg_validation_pre2 + avg_validation_pre3 + avg_validation_pre4 + avg_validation_pre5)/5
                avg_validation_pre = total_validation_pre_value/itr_validation

                avg_validation_rec1 = total_validation_rec_value1/itr_validation
                avg_validation_rec2 = total_validation_rec_value2/itr_validation
                avg_validation_rec3 = total_validation_rec_value3/itr_validation
                avg_validation_rec4 = total_validation_rec_value4/itr_validation
                avg_validation_rec5 = total_validation_rec_value5/itr_validation
                #avg_validation_rec = (avg_validation_rec1 + avg_validation_rec2 + avg_validation_rec3 + avg_validation_rec4 + avg_validation_rec5)/5
                avg_validation_rec = total_validation_rec_value/itr_validation

                avg_validation_miou1 = total_validation_miou_value1/itr_validation
                avg_validation_miou2 = total_validation_miou_value2/itr_validation
                avg_validation_miou3 = total_validation_miou_value3/itr_validation
                avg_validation_miou4 = total_validation_miou_value4/itr_validation
                avg_validation_miou5 = total_validation_miou_value5/itr_validation
                #avg_validation_miou = (avg_validation_miou1 + avg_validation_miou2 + avg_validation_miou3 + avg_validation_miou4 + avg_validation_miou5)/5
                avg_validation_miou = total_validation_miou_value/itr_validation

                avg_auc_v1 = total_auc_v1/itr_validation
                avg_auc_v2 = total_auc_v2/itr_validation
                avg_auc_v3 = total_auc_v3/itr_validation
                avg_auc_v4 = total_auc_v4/itr_validation
                avg_auc_v5 = total_auc_v5/itr_validation
                avg_auc_v = total_auc_v/itr_validation

                avg_validation_dice1 = total_validation_dice_value1/iteration_no
                avg_validation_dice2 = total_validation_dice_value2/iteration_no
                avg_validation_dice3 = total_validation_dice_value3/iteration_no
                avg_validation_dice4 = total_validation_dice_value4/iteration_no
                avg_validation_dice5 = total_validation_dice_value5/iteration_no
                #avg_validation_dice = (avg_validation_dice1 + avg_validation_dice2 + avg_validation_dice3 + avg_validation_dice4 + avg_validation_dice5)/5
                avg_validation_dice = total_validation_dice_value/iteration_no

                avg_validation_mean_class_acc = total_validation_mean_class_acc_value/itr_validation

                avg_validation_acc1 = total_validation_acc_value1/itr_validation
                avg_validation_acc2 = total_validation_acc_value2/itr_validation
                avg_validation_acc3 = total_validation_acc_value3/itr_validation
                avg_validation_acc4 = total_validation_acc_value4/itr_validation
                avg_validation_acc5 = total_validation_acc_value5/itr_validation
                #avg_validation_acc = (avg_validation_acc1 + avg_validation_acc2 + avg_validation_acc3 + avg_validation_acc4 + avg_validation_acc5)/5
                avg_validation_acc = total_validation_acc_value/itr_validation

                avg_validation_pix_acc1 = total_validation_pix_acc_value1/itr_validation
                avg_validation_pix_acc2 = total_validation_pix_acc_value2/itr_validation
                avg_validation_pix_acc3 = total_validation_pix_acc_value3/itr_validation
                avg_validation_pix_acc4 = total_validation_pix_acc_value4/itr_validation
                avg_validation_pix_acc5 = total_validation_pix_acc_value5/itr_validation
                avg_validation_pix_acc = (avg_validation_pix_acc1 + avg_validation_pix_acc2 + avg_validation_pix_acc3 + avg_validation_pix_acc4 + avg_validation_pix_acc5)/5

                end = time.time()
                duration = end - start
                #print("     Validation Iteration: {}/{} | {:.2f}s  - Loss: {:.8f}%, Accuracy: {:.8f}%, Precision: {:.8f}%, Recall: {:.8f}%, PixelAccuracy: {:.8f}%, MeanClassAccuracy: {:.3f}%, Mean_IoU: {:.3f}%".format(itr_validation,
                #itr_validation, duration, avg_validation_loss * 100, avg_validation_acc * 100, avg_validation_pre * 100, avg_validation_rec * 100, avg_validation_pix_acc * 100, avg_validation_mean_class_acc * 100, avg_validation_miou * 100))
                #if avg_validation_acc > avg_accuracy:
                print("[{}/{}] | {:.2f}s - ".format(iteration_no, iteration_no, duration))
                print("     Train- loss: {:.6f}%, pre: {:.6f}%, rec: {:.6f}%, m_iou: {:.4f}%, dice: {:.4f}%, m_acc: {:.3f}%, pix_acc: {:.4f}%, acc: {:.4f}%, auc: {:.4f}%".format(avg_loss * 100,
                avg_precision * 100, avg_recall * 100, avg_miou * 100, avg_dice * 100, avg_mean_class_acc * 100, avg_pix_acc * 100, avg_accuracy * 100, avg_auc * 100))
                print("     Train- pre1: {:.4f}%, pre2: {:.4f}%, pre3: {:.4f}%, pre4: {:.4f}%, pre5: {:.4f}%".format(avg_precision1 * 100, avg_precision2 * 100, avg_precision3 * 100, avg_precision4 * 100, avg_precision5 * 100))
                print("     Train- rec1: {:.4f}%, rec2: {:.4f}%, rec3: {:.4f}%, rec4: {:.4f}%, rec5: {:.4f}%".format(avg_recall1 * 100, avg_recall2 * 100, avg_recall3 * 100, avg_recall4 * 100, avg_recall5 * 100))
                print("     Train- m_iou1: {:.3f}%, m_iou2: {:.3f}%, m_iou3: {:.3f}%, m_iou4: {:.3f}% m_iou5: {:.3f}%".format(avg_miou1 * 100, avg_miou2 * 100, avg_miou3 * 100, avg_miou4 * 100, avg_miou5 * 100))
                print("     Train- dice1: {:.3f}%, dice2: {:.3f}%, dice3: {:.3f}%, dice4: {:.3f}%, dice5: {:.3f}%".format(avg_dice1 * 100, avg_dice2 * 100, avg_dice3 * 100, avg_dice4 * 100, avg_dice5 * 100))
                print("     Train- auc1: {:.3f}%, auc2: {:.3f}%, auc3: {:.3f}%, auc4: {:.3f}%, auc5: {:.3f}%".format(avg_auc1 * 100, avg_auc2 * 100, avg_auc3 * 100, avg_auc4 * 100, avg_auc5 * 100))
                print("[{}/{}] |".format(itr_validation, itr_validation))
                print("     Valid- loss: {:.6f}%, pre: {:.6f}%, rec: {:.6f}%, m_iou: {:.4f}%, dice: {:.4f}%, m_acc: {:.3f}%, pix_acc: {:.4f}%, acc: {:.4f}%, auc: {:.4f}%".format(avg_validation_loss * 100,
                avg_validation_pre * 100, avg_validation_rec * 100, avg_validation_miou * 100, avg_validation_dice * 100, avg_validation_mean_class_acc * 100, avg_validation_pix_acc * 100, avg_validation_acc * 100, avg_auc_v * 100))

                '''
                print("     acc: {:.4f}%, acc1: {:.4f}%, acc2: {:.4f}%, acc3: {:.4f}%, acc4: {:.4f}%, acc5: {:.4f}%, val_acc: {:.4f}%".format(avg_accuracy * 100,
                avg_accuracy1 * 100, avg_accuracy2 * 100, avg_accuracy3 * 100, avg_accuracy4 * 100, avg_accuracy5 * 100, avg_validation_acc * 100))

                print("     pix_acc: {:.4f}%, val_pix_acc: {:.4f}%, pix_acc1: {:.4f}%, pix_acc2: {:.4f}%, pix_acc3: {:.4f}%, m_acc: {:.4f}%".format(avg_pix_acc * 100,
                avg_validation_pix_acc * 100, avg_pix_acc1 * 100, avg_pix_acc2 * 100, avg_pix_acc3 * 100, avg_mean_class_acc * 100))
                '''

                print("tp1: {:.2f}, tp2: {:.2f}, tp3: {:.2f}, tp4: {:.2f}, tp5: {:.2f}, fp1: {:.2f}, fp2: {:.2f}, fp3: {:.2f}, fp4: {:.2f}, fp5: {:.2f}, fn1: {:.2f}, fn2: {:.2f}, fn3: {:.2f}, fn4: {:.2f}, fn5: {:.2f}".format(tp1,
                tp2, tp3, tp4, tp5, fp1, fp2, fp3, fp4, fp5, fn1, fn2, fn3, fn4, fn5))

                '''
                if avg_validation_miou < best_miou:
                    best_miou = avg_validation_miou
                    save_path = saver.save(sess, MODEL_DIR + "model.ckpt")
                '''

                if avg_validation_loss < best_val_loss:
                    best_val_loss = avg_validation_loss
                    #print("     validation loss improved to {:.8f}%".format(best_val_loss * 100))
                    #print("     validation loss improved to {:.8f}% | model is saving...".format(best_val_loss * 100))
                    save_path = saver.save(sess, MODEL_DIR + "model.ckpt")
                    #print("     ...model update!!!")
                    non_update_counter = 0
                else:
                    non_update_counter += 1
                    #print("     validation loss did not improve for the last {} epochs...".format(non_update_counter))

                if train_model_update_counter == 0:
                    print("     training loss improved to {:.8f}% | model is saving...".format(best_train_loss * 100))
                else:
                    print("     training loss did not improve for the last {} epochs...".format(train_model_update_counter))

                if non_update_counter == 0:
                    print("     validation loss improved to {:.8f}%".format(best_val_loss * 100))
                else:
                    print("     validation loss did not improve for the last {} epochs...".format(non_update_counter))

            #with tf.Graph().as_default(), tf.device('/gpu:0'):
            print('Testing Starts...')
            itr_testing_no = int(len(TEST_SET)/BATCH_SIZE)
            for t in range(0, int(len(TEST_SET)/BATCH_SIZE)):
                #print('test batch: ', str(t + 1))
                test_set = []
                for tsample in range(0, BATCH_SIZE):
                    test_set.append(read_sample_data(TEST_SET[(t * BATCH_SIZE) + tsample]))

                #print('test_set: ', str(len(test_set)))
                for o in range(len(test_set)):
                    data_row = test_set[o][0]
                    TEST_FEATURES.append([data_row[0].todense(), data_row[1].todense(), data_row[2].todense(), data_row[3].todense(), data_row[4].todense()])
                    TEST_LABELS.append([data_row[5].todense(), data_row[6].todense(), data_row[7].todense(), data_row[8].todense(), data_row[9].todense()])
                test_data = np.asarray(TEST_FEATURES, dtype=np.float32)
                test_y_true = np.asarray(TEST_LABELS, dtype=np.float32)
                #print('e_labels.shape: ', e_labels.shape)
                x1, x2, x3, x4, x5 = reshape_input(test_data)
                y1, y2, y3, y4, y5 = reshape_input(test_y_true)
                #print('y1.shape: ', y1.shape)
                #print('y1 :', y1[0])
                y1 = np.reshape(y1, [y1.shape[0], y1.shape[1], y1.shape[2]])
                #print(y1.shape)
                y2 = np.reshape(y2, [y2.shape[0], y2.shape[1], y2.shape[2]])
                #print(y2.shape)
                y3 = np.reshape(y3, [y3.shape[0], y3.shape[1], y3.shape[2]])
                #print(y3.shape)
                y4 = np.reshape(y4, [y4.shape[0], y4.shape[1], y4.shape[2]])
                #print(y3.shape)
                y5 = np.reshape(y5, [y5.shape[0], y5.shape[1], y5.shape[2]])
                #print(y3.shape)

                #print('y1 :', y1[0])

                feed = {x1_placeholder: x1, x2_placeholder: x2, x3_placeholder: x3,  x4_placeholder: x4,  x5_placeholder: x5, mode:'test'}
                #feed = {x1_placeholder: x1, x2_placeholder: x2, x3_placeholder: x3, y_placeholder: e_labels, mode:'test'}

                saver.restore(sess, MODEL_DIR + "model.ckpt")
                print("Model restored from file: %s" % save_path)
                pred_op1, pred_op2, pred_op3, pred_op4, pred_op5 = sess.run([pred1, pred2, pred3, pred4, pred5], feed_dict= feed)
                #print('pred_op1: ', pred_op1)
                #print('pred_op1: ', len(pred_op1[0]))
                #pred_op, pix_acc, class_acc, miou, acc, summary_value_test = sess.run([pred, pixel_accuracy, mean_class_accuracy, mean_iou, accuracy_op, summary_op], feed_dict= feed)
                #print("Testing => Batch: {}/{}, Accuracy: {:.8f}%, PixelAccuracy: {:.4f}%, MeanClassAccuracy: {:.3f}%, Mean_IoU: {:.3f}%".format(t+1,
                #itr_testing_no, acc * 100, pix_acc * 100, class_acc * 100, miou * 100))

                TEST_FEATURES = []
                TEST_LABELS = []
                res = []
                b = 0
                threshold_value = 0.45
                for cnt in range(0, BATCH_SIZE):
                    y_t = []
                    y_t.append(y1[cnt])
                    y_t.append(y2[cnt])
                    y_t.append(y3[cnt])
                    y_t.append(y4[cnt])
                    y_t.append(y5[cnt])

                    p_t = []
                    p_t.append(pred_op1[cnt])
                    p_t.append(pred_op2[cnt])
                    p_t.append(pred_op3[cnt])
                    p_t.append(pred_op4[cnt])
                    p_t.append(pred_op5[cnt])

                    for yf in range(0, NO_OF_FRAMES):
                        fontsz = "14"
                        fig = plt.figure(figsize=(14, 14))
                        fig.canvas.set_window_title('Predicted sequence for ground labels: Batch_ ' + str(t) + '_' + str(cnt+1)+ '_'+ str(yf))
                        fig.add_subplot(2, 1, 1)
                        plt.xlabel("target label grid",fontsize="14")
                        plt.matshow(y_t[yf].T, fignum=False)
                        label_frame = y_t[yf]
                        cnt_1 = label_frame[np.where(label_frame == 1.0)]
                        print('label (No. of cars): ', len(cnt_1))
                        frame1 = np.reshape(p_t[yf], (450, 100))
                        frame1_ = np.reshape(p_t[yf], (450, 100))
                        sorted_frame = sorted(frame1_[np.nonzero(frame1_)])
                        print('pred label for cars: ', sorted_frame[-(len(cnt_1)+5):])
                        print('pred label> 1:', np.count_nonzero(frame1))
                        print('pred label> 0:', np.count_nonzero(frame1==0))

                        frame1 = compute_threshold(frame1, threshold_value)
                        fig.add_subplot(2, 1, 2)
                        plt.xlabel("predicted label grid",fontsize="14")
                        plt.matshow(frame1.T, fignum=False)
                        cnt_1 = frame1[np.where(frame1 == 1.0)]
                        print('pred - 1: ', cnt_1)
                        print('pred 1 count: ', len(cnt_1))

                        fig.savefig(os.path.join(MODEL_DIR + 'test/' , 'pred_image_'+ str(t) + "_" + str(cnt) + '_'+ str(yf) + ".png"), dpi=200)
                        plt.show()

                        print('saving image: ' + 'figure_frame_'+ str(t) + "_" + str(cnt) + '_'+ str(yf) + ".png")

                    '''
                    fontsz = "14"
                    fig = plt.figure(figsize=(16, 8))
                    fig.canvas.set_window_title('Predicted sequence for ground labels: Batch_ ' + str(t) + '_' + str(cnt+1))
                    fig.add_subplot(5, 2, 1)
                    plt.xlabel("target y1 grid",fontsize=fontsz)
                    plt.matshow(y1[cnt].T, fignum=False)
                    label_frame1 = y1[cnt]
                    cnt_1 = label_frame1[np.where(label_frame1 == 1.0)]

                    fig.add_subplot(5, 2, 3)
                    plt.xlabel("target y2 grid",fontsize=fontsz)
                    plt.matshow(y2[cnt].T, fignum=False)
                    label_frame2 = y2[cnt]
                    cnt_2 = label_frame2[np.where(label_frame2 == 1.0)]

                    fig.add_subplot(5, 2, 5)
                    plt.xlabel("target y3 grid",fontsize=fontsz)
                    plt.matshow(y3[cnt].T, fignum=False)
                    label_frame3 = y3[cnt]
                    cnt_3 = label_frame3[np.where(label_frame3 == 1.0)]

                    fig.add_subplot(5, 2, 7)
                    plt.xlabel("target y4 grid",fontsize=fontsz)
                    plt.matshow(y4[cnt].T, fignum=False)
                    label_frame4 = y4[cnt]
                    cnt_4 = label_frame4[np.where(label_frame4 == 1.0)]

                    fig.add_subplot(5, 2, 9)
                    plt.xlabel("target y5 grid",fontsize=fontsz)
                    plt.matshow(y5[cnt].T, fignum=False)
                    label_frame5 = y5[cnt]
                    cnt_5 = label_frame5[np.where(label_frame5 == 1.0)]

                    print('label1 (No. of cars): ', len(cnt_1))
                    frame1 = np.reshape(pred_op1[cnt], (450, 100))
                    frame1_ = np.reshape(pred_op1[cnt], (450, 100))
                    sorted_frame = sorted(frame1_[np.nonzero(frame1_)])
                    print('cars pred probability for label1: ', sorted_frame[-len(cnt_1)+5:])
                    print('pred label> 1:', np.count_nonzero(frame1))
                    print('pred label> 0:', np.count_nonzero(frame1==0))

                    print('label2 (No. of cars): ', len(cnt_2))
                    frame2 = np.reshape(pred_op2[cnt], (450, 100))
                    frame2_ = np.reshape(pred_op2[cnt], (450, 100))
                    sorted_frame = sorted(frame2_[np.nonzero(frame2_)])
                    print('cars pred probability for label2: ', sorted_frame[-len(cnt_2)+5:])
                    print('pred label> 1:', np.count_nonzero(frame2))
                    print('pred label> 0:', np.count_nonzero(frame2==0))

                    #print('label3 (No. of cars): ', len(cnt_3))
                    frame3 = np.reshape(pred_op3[cnt], (450, 100))
                    frame3_ = np.reshape(pred_op3[cnt], (450, 100))
                    sorted_frame = sorted(frame3_[np.nonzero(frame3_)])
                    print('cars pred probability for label3: ', sorted_frame[-len(cnt_3)+5:])
                    print('pred label> 1:', np.count_nonzero(frame3))
                    print('pred label> 0:', np.count_nonzero(frame3==0))

                    #print('label4 (No. of cars): ', len(cnt_4))
                    frame4 = np.reshape(pred_op4[cnt], (450, 100))
                    frame4_ = np.reshape(pred_op4[cnt], (450, 100))
                    sorted_frame = sorted(frame4_[np.nonzero(frame4_)])
                    print('cars pred probability for label4: ', sorted_frame[-len(cnt_4)+5:])
                    print('pred label> 1:', np.count_nonzero(frame4))
                    print('pred label> 0:', np.count_nonzero(frame4==0))

                    #print('label5 (No. of cars): ', len(cnt_5))
                    frame5 = np.reshape(pred_op5[cnt], (450, 100))
                    frame5_ = np.reshape(pred_op5[cnt], (450, 100))
                    sorted_frame = sorted(frame5_[np.nonzero(frame5_)])
                    print('cars pred probability for label5: ', sorted_frame[-len(cnt_5)+5:])
                    print('pred label> 1:', np.count_nonzero(frame5))
                    print('pred label> 0:', np.count_nonzero(frame5==0))

                    frame1 = compute_threshold(frame1, threshold_value)
                    fig.add_subplot(5, 2, 2)
                    plt.xlabel("predicted y1 grid",fontsize=fontsz)
                    plt.matshow(frame1.T, fignum=False)
                    cnt_ = frame1[np.where(frame1 == 1.0)]
                    print('pred label1 count 1: ', cnt_)
                    print('pred label1 1 len: ', len(cnt_))

                    frame2 = compute_threshold(frame2, threshold_value)
                    fig.add_subplot(5, 2, 4)
                    plt.xlabel("predicted y2 grid",fontsize=fontsz)
                    plt.matshow(frame2.T, fignum=False)
                    cnt_ = frame2[np.where(frame2 == 1.0)]
                    print('pred label2 count 1: ', cnt_)
                    print('pred label2 len: ', len(cnt_))

                    frame3 = compute_threshold(frame3, threshold_value)
                    fig.add_subplot(5, 2, 6)
                    plt.xlabel("predicted y3 grid",fontsize=fontsz)
                    plt.matshow(frame3.T, fignum=False)
                    cnt_ = frame3[np.where(frame3 == 1.0)]
                    print('pred label3 count 1: ', cnt_)
                    print('pred label3 len: ', len(cnt_))

                    frame4 = compute_threshold(frame4, threshold_value)
                    fig.add_subplot(5, 2, 8)
                    plt.xlabel("predicted y4 grid",fontsize=fontsz)
                    plt.matshow(frame4.T, fignum=False)
                    cnt_ = frame4[np.where(frame4 == 1.0)]
                    print('pred label4 count 1: ', cnt_)
                    print('pred label4 len: ', len(cnt_))

                    frame5 = compute_threshold(frame5, threshold_value)
                    fig.add_subplot(5, 2, 10)
                    plt.xlabel("predicted y5 grid",fontsize=fontsz)
                    plt.matshow(frame5.T, fignum=False)
                    cnt_ = frame5[np.where(frame5 == 1.0)]
                    print('pred label5 count 1: ', cnt_)
                    print('pred label5 len: ', len(cnt_))

                    plt.tight_layout()

                    #fig.subplots_adjust(wspace=0.1, hspace=0.1)

                    #misc.imsave(os.path.join(MODEL_DIR + 'test/' , 'pred_image_'+ str(t) + "_" + str(cnt)+ ".png"), frame1)
                    fig.savefig(os.path.join(MODEL_DIR + 'test/' , 'pred_image_'+ str(t) + "_" + str(cnt)+ ".png"), dpi=200)
                    plt.show()

                    print('saving image: ' + 'figure_frame_'+ str(t) + "_" + str(cnt)+ ".png")
                    '''

if __name__ == "__main__":
  tf.app.run()


'''
TRAIN_SAMPLES = int(0.8 * TOTAL_SAMPLES)
VAL_TEST_SAMPLES = TOTAL_SAMPLES - TRAIN_SAMPLES
if int((TOTAL_SAMPLES * 0.1) % BATCH_SIZE) != 0:
    VALIDATION_SAMPLES = int(VAL_TEST_SAMPLES * 0.5) + 2
    TEST_SAMPLES = int(VAL_TEST_SAMPLES * 0.5) - 2
else:
    VALIDATION_SAMPLES = int(0.1 * TOTAL_SAMPLES)
    TEST_SAMPLES = int(0.1 * TOTAL_SAMPLES)

def conv_res_block(input, filters, kernel, stride, block, identity_connection=True):
    filter1 = filters
    filter2 = filters * 4
    conv_name_base = 'res_' + block + '_branch'
    bn_name_base = 'bn_' + block + '_branch'

    #branch1
    if not identity_connection:
        if block == 'x1_3a' or block == 'x2_3a' or block == 'x3_3a' or block == 'x4_3a' or block == 'x5_3a':
            shortcut = tf.layers.conv2d(inputs=input, filters=filter2, kernel_size=[1, 2], strides=(stride, stride), padding='valid', name=conv_name_base+'1')
        else:
            shortcut = tf.layers.conv2d(inputs=input, filters=filter2, kernel_size=[1, 1], strides=(stride, stride), padding='same', name=conv_name_base+'1')
        shortcut = tf.layers.batch_normalization(shortcut, axis=-1, training=True, name=bn_name_base+'1')
    else:
        shortcut = input

    #branch2
    if block == 'x1_3a' or block == 'x2_3a' or block == 'x3_3a' or block == 'x4_3a' or block == 'x5_3a':
        x = tf.layers.conv2d(inputs=input, filters=filter1, kernel_size=[1, 2], strides=(stride, stride), padding='valid', name=conv_name_base+'2a')
    else:
        x = tf.layers.conv2d(inputs=input, filters=filter1, kernel_size=[1, 1], strides=(stride, stride), padding='same', name=conv_name_base+'2a')
    x = tf.layers.batch_normalization(x, axis=-1, training=True, name=bn_name_base+'2a')
    #x = tf.nn.relu(x, name=conv_name_base+'2a_relu')
    x = tf.nn.elu(x, name=conv_name_base+'2a_elu')

    x = tf.layers.conv2d(inputs=x, filters=filter1, kernel_size=[kernel, kernel], strides=(1, 1), padding='same', name=conv_name_base+'2b')
    x = tf.layers.batch_normalization(x, axis=-1, training=True, name=bn_name_base+'2b')
    #x = tf.nn.relu(x, name=conv_name_base+'2b_relu')
    x = tf.nn.elu(x, name=conv_name_base+'2b_elu')

    x = tf.layers.conv2d(inputs=x, filters=filter2, kernel_size=[1, 1], strides=(1, 1), padding='same', name=conv_name_base+'2c')
    x = tf.layers.batch_normalization(x, axis=-1, training=True, name=bn_name_base+'2c')

    #add
    x = tf.add(x, shortcut, name='res' + block)
    #x = tf.nn.relu(x, name='res' + block + '_relu')
    x = tf.nn.elu(x, name='res' + block + '_elu')
    return x

def encoder_resnet(x, seq, mode):
    with tf.device('/gpu:0'):
        #config = tf.ConfigProto(allow_soft_placement=True)
        training = tf.equal(mode, 'train')
        conv_x = conv2d_layer(x, FILTERS[0], KERNEL_SIZE, 1, PADDING_SAME, seq)  #[batch_size, 450, 100, 8]
        print('conv_x: ', conv_x)
        # ResNet101 as a backbone structure for encoder
        # strating block
        conv1_x = start_conv_block(conv_x, FILTERS[0], KERNEL_SIZE, 2, seq) # [batch_size, 225, 50, 8]
        print("after x start conv block:", conv1_x.shape)
        # pool block
        pool_x = start_pool_block(conv1_x, KERNEL_SIZE, 2, seq) # [batch_size, 112, 25, 8]
        print("after x pool block:", pool_x.shape)
        # residual block1
        conv2_x = conv_res_block(pool_x, FILTERS[0], KERNEL_SIZE, 1, 'x' + str(seq) + '_2a', identity_connection=False) # [batch_size, 112, 25, 8]
        conv2_x = conv_res_block(conv2_x, FILTERS[0], KERNEL_SIZE, 1, 'x' + str(seq) + '_2b')
        conv2_x = conv_res_block(conv2_x, FILTERS[0], KERNEL_SIZE, 1, 'x' + str(seq) + '_2c')
        print("after x block1:", conv2_x.shape)
        # residual block2
        conv3_x = conv_res_block(conv2_x, FILTERS[1], KERNEL_SIZE, 2, 'x' + str(seq) + '_3a', identity_connection=False) # [batch_size, 56, 12, 16]
        for i in range(1, 4):
            conv3_x = conv_res_block(conv3_x, FILTERS[1], KERNEL_SIZE, 1, 'x' + str(seq) + '_3b_' + str(i))
        print("after x block2:", conv3_x.shape)
        # residual block3
        conv4_x = conv_res_block(conv3_x, FILTERS[2], KERNEL_SIZE, 2, 'x' + str(seq) + '_4a', identity_connection=False) # [batch_size, 28, 6, 32]
        for i in range(1, 23):
            conv4_x = conv_res_block(conv4_x, FILTERS[2], KERNEL_SIZE, 1, 'x' + str(seq) + '_4b_' + str(i))
        print("after x block3:", conv4_x.shape)
        # residual block4
        conv5_x = conv_res_block(conv4_x, FILTERS[3], KERNEL_SIZE, 2, 'x' + str(seq) + '_5a', identity_connection=False) # [batch_size, 14, 3, 64]
        conv5_x = conv_res_block(conv5_x, FILTERS[3], KERNEL_SIZE, 1, 'x' + str(seq) + '_5b')
        conv5_x = conv_res_block(conv5_x, FILTERS[3], KERNEL_SIZE, 1, 'x' + str(seq) + '_5c')
        print("after x block4:", conv5_x.shape)

        return conv5_x, conv4_x, conv3_x, conv2_x, conv1_x, conv_x

def decoder_upsampling(lstm_output, conv4_f, conv3_f, conv2_f, conv1_f, conv_f, seq, mode):
    with tf.device('/gpu:0'):
        #config = tf.ConfigProto(allow_soft_placement=True)
        conv_feat = tf.layers.conv2d(inputs=lstm_output, filters=RES_OUT_FILTERS[3], kernel_size=[1, 1], strides=(1, 1),  padding='same', activation=tf.nn.elu)
        print('conv_feat: ', conv_feat)
        up1 = tf.layers.conv2d_transpose(inputs=conv_feat, filters=RES_OUT_FILTERS[2], kernel_size=[3, 3], padding="same", strides=2, activation=tf.nn.elu) # [batch_size, 28, 6, 32]
        print('up1: ', up1)
        skip_1 = tf.add(up1, conv4_f, name="skip_1")
        print('skip_1: ', skip_1)
        conv6 = conv2d_layer(skip_1, RES_OUT_FILTERS[2], 1, 1, PADDING_SAME, seq+10)
        print('conv6: ', conv6)

        up2 = tf.layers.conv2d_transpose(inputs=conv6, filters=RES_OUT_FILTERS[1], kernel_size=[3, 3], padding="same", strides=2, activation=tf.nn.elu) # [batch_size, 56, 12, 16]
        print('up2: ', up2)
        skip_2 = tf.add(up2, conv3_f, name="skip_2")
        print('skip_2: ', skip_2)
        conv7 = conv2d_layer(skip_2, RES_OUT_FILTERS[1], 1, 1, PADDING_SAME, seq+20)
        print('conv7: ', conv7)

        up3 = tf.layers.conv2d_transpose(inputs=conv7, filters=RES_OUT_FILTERS[0], kernel_size=[2, 3], padding="valid", strides=2, activation=tf.nn.elu) # [batch_size, 112, 25, 8]
        print('up3: ', up3)
        skip_3 = tf.add(up3, conv2_f, name="skip_3")
        print('skip_3: ', skip_3)
        conv8 = conv2d_layer(skip_3, RES_OUT_FILTERS[0], 1, 1, PADDING_SAME, seq+30)
        print('conv8: ', conv8)

        up4 = tf.layers.conv2d_transpose(inputs=conv8, filters=FILTERS[0], kernel_size=[3, 2], padding="valid", strides=2, activation=tf.nn.elu) # [batch_size, 225, 50, 8]
        print('up4: ', up4)
        skip_4 = tf.add(up4, conv1_f, name="skip_4")
        print('skip_4: ', skip_4)
        conv9 = conv2d_layer(up4, FILTERS[0], 1, 1, PADDING_SAME, seq+40)
        print('conv9: ', conv9)

        up5 = tf.layers.conv2d_transpose(inputs=conv9, filters=FILTERS[0], kernel_size=[3, 3], padding="same", strides=2, activation=tf.nn.elu) #[batch_size, 450, 100, 8]
        print('up5: ', up5)
        skip_5 = tf.add(up5, conv_f, name="skip_5")
        print('skip_5: ', skip_5)
        conv10 = conv2d_layer(up5, FILTERS[0], 1, 1, PADDING_SAME, seq+50)
        print('conv10: ', conv10)

        logits = tf.layers.conv2d(inputs=conv10, filters=1, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=(1, 1), padding='same', activation=tf.nn.elu)
        print('logits: ', logits)

        pred = tf.layers.conv2d(inputs=logits, filters=1, kernel_size=[1, 1], strides=(1, 1), padding='same', activation=tf.nn.sigmoid)
        print('predictions: ', pred)
        return logits, pred


    '''
