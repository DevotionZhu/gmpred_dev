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
from configuration import Configuration
from sklearn.utils.class_weight import compute_class_weight

tf.logging.set_verbosity(tf.logging.INFO)

# for 1ts
DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_4frames_1ts/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_1ts_wbce/"
MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_1ts_wbce_40ep/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_1ts_bd/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_1ts_bd_v/"

# for 2ts
#DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_4frames_2ts/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_2ts_wbce/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_2ts_wbce_50ep/"

# for 3ts
#DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_4frames_3ts/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_3ts_bce/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_3ts_wbce_50ep/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_3ts_bd/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_recur_model_wo_skiplstm_3ts_bd_v/"

# for 5ts
#DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_4frames_5ts/"
#MODEL_DIR = "/home/mushfiqrahman/dataset/db_numpy/ds/convlstm_resnet_unet_model_5ts_bd/"

BATCH_SIZE = 4
EPOCHS = 40
LEARNING_RATE = 0.0001
#TRAINING_STEPS = 1000
THRESHOLD_ARRAYS = []
POOL_SIZE  = 2
KERNEL_SIZE = 3
PADDING_SAME = "same"
PADDING_VALID = "valid"
LABELS_PIXEL_1_COUNT = 0
TRAIN_SET = []
VALIDATION_SET = []
TEST_SET = []
eps = 1e-8
NUM_CLASSES = 2 # {0 = background, 1 = car}
NO_OF_FRAMES = 3
INPUT_HEIGTH = 450
INPUT_WIDTH = 100
FILTERS = [8, 16, 32, 64, 128, 256, 512]
RES_FILTERS = [32, 64, 128, 256, 512, 1024]
ratio = 30 / 45000
ALPHA = 0.5
BETA = 0.99

# 1 time step ahead frame prediction
TOTAL_SAMPLES = 4776
TRAIN_SAMPLES = 4000
VALIDATION_SAMPLES = 500
TEST_SAMPLES = 276

'''
# 2 time step ahead frame prediction
TOTAL_SAMPLES = 4740
TRAIN_SAMPLES = 4000
VALIDATION_SAMPLES = 500
TEST_SAMPLES = 240
'''

'''
# 3 time step ahead frame prediction
TOTAL_SAMPLES = 4704
TRAIN_SAMPLES = 4000
VALIDATION_SAMPLES = 500
TEST_SAMPLES = 204
'''

'''
# 5 time step ahead frame prediction
TOTAL_SAMPLES = 4632
TRAIN_SAMPLES = 4000
VALIDATION_SAMPLES = 500
TEST_SAMPLES = 132
'''

def reshape_input(x):
    x1 = []
    x2 = []
    x3 = []
    for i in range(BATCH_SIZE):
        #print('i elem>:', str(i))
        #print(input_arr[i][0])
        x1.append(x[i][0])
        x2.append(x[i][1])
        x3.append(x[i][2])
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    #print(x1.shape)
    x1 = np.reshape(x1, [x1.shape[0], x1.shape[1], x1.shape[2], 1])
    #print(x1.shape)
    x2 = np.reshape(x2, [x2.shape[0], x2.shape[1], x2.shape[2], 1])
    #print(x2.shape)
    x3 = np.reshape(x3, [x3.shape[0], x3.shape[1], x3.shape[2], 1])
    #print(x3.shape)

    return x1, x2, x3

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
        if block == 'x1_3a' or block == 'x2_3a' or block == 'x3_3a':
            shortcut = tf.layers.conv2d(inputs=input, filters=filter2, kernel_size=[1, 2], strides=(stride, stride), padding='valid', name=conv_name_base+'1')
        elif block == 'x1_4a' or block == 'x2_4a' or block == 'x3_4a':
            shortcut = tf.layers.conv2d(inputs=input, filters=filter2, kernel_size=[1, 1], strides=(stride, stride), padding='valid', name=conv_name_base+'1')
        else:
            shortcut = tf.layers.conv2d(inputs=input, filters=filter2, kernel_size=[1, 1], strides=(stride, stride), padding='same', name=conv_name_base+'1')
        shortcut = tf.layers.batch_normalization(shortcut, axis=-1, training=True, name=bn_name_base+'1')
    else:
        shortcut = input

    #branch2
    if block == 'x1_3a' or block == 'x2_3a' or block == 'x3_3a':
        x = tf.layers.conv2d(inputs=input, filters=filter1, kernel_size=[1, 2], strides=(stride, stride), padding='valid', name=conv_name_base+'2a')
    elif block == 'x1_4a' or block == 'x2_4a' or block == 'x3_4a':
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
    print('hidden_state :', hidden_state)
    convlstm_output = tf.split(convlstm_output, tf.ones((1), dtype=tf.int32 ), 1)
    #print('convlstm_output: ', convlstm_output)
    convlstm_output_last = convlstm_output[-1]
    convlstm_output_last = tf.reshape(convlstm_output_last, [BATCH_SIZE, input_shape[0], input_shape[1], input_shape[2]])
    #print('convlstm_output_last: ', convlstm_output_last)
    return convlstm_output_last, hidden_state

def convlstm_block(input_shape, input_stack, kernel, init_state, name):
    #with tf.variable_scope(name):
    stack_frames = tf.stack([input_stack[0], input_stack[1], input_stack[2]], axis=1)
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
    convlstm_output = tf.split(convlstm_output, tf.ones((NO_OF_FRAMES), dtype=tf.int32), 1)
    #print('convlstm_output: ', convlstm_output)
    convlstm_output_last = convlstm_output[-1]
    convlstm_output_last = tf.reshape(convlstm_output_last, [BATCH_SIZE, input_shape[0], input_shape[1], input_shape[2]])
    #print('convlstm_output_last: ', convlstm_output_last)

    return convlstm_output_last, hidden_state

def recurrent_skip(y, state, seq_name):
    feat_shape = y.get_shape().as_list()[1:]
    output1, state1  = skip_convlstm_block(feat_shape, y, KERNEL_SIZE, state, seq_name)
    #print('skip_convlstm_output: ', state1)
    return output1, state1

def encoder_net(x, frameno, mode):
    with tf.device('/gpu:0'):
        training = tf.equal(mode, 'train')
        # ResNet101 as a backbone structure for encoder
        # strating block
        # 1st conv + relu + bn + convlstm block
        conv1_x = start_conv_block(x, FILTERS[1], KERNEL_SIZE, 2, str(frameno))  #[batch_size, 225, 50, 16]
        print('conv1_x: ', conv1_x.shape)
        #convlstm1_o, convlstm1_s = recurrent_skip(conv1_x, convlstm1_s, 'convlstm_skip1_' + str(frameno))

        pool1_x = start_pool_block(conv1_x, (KERNEL_SIZE, 2), (2,2), str(frameno))  #[batch_size, 112, 25, 16]
        print('pool1_x: ', pool1_x.shape)

        # residual block1
        conv2_x = conv_res_block(pool1_x, FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frameno)+'_2a', identity_connection=False) # [batch_size, 112, 25, 64]
        conv2_x = conv_res_block(conv2_x, FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frameno)+'_2b')
        conv2_x = conv_res_block(conv2_x, FILTERS[1], KERNEL_SIZE, 1, 'x'+str(frameno)+'_2c')
        print("BLOCK1 - X" + str(frameno)+": ", conv2_x.shape)
        #convlstm2_o, convlstm2_s = recurrent_skip(conv2_x, convlstm2_s, 'convlstm_skip2_' + str(frameno))

        # residual block2
        conv3_x = conv_res_block(conv2_x, FILTERS[2], KERNEL_SIZE, 2, 'x'+str(frameno)+'_3a', identity_connection=False) # [batch_size, 56, 12, 128]
        for r in range(1, 4):
            conv3_x = conv_res_block(conv3_x, FILTERS[2], KERNEL_SIZE, 1, 'x'+str(frameno)+'_3b'+str(r))
        print("BLOCK2 - X" + str(frameno)+": ", conv3_x.shape)
        #convlstm3_o, convlstm3_s = recurrent_skip(conv3_x, convlstm3_s, 'convlstm_skip3_' + str(frameno))

        # residual block3
        conv4_x = conv_res_block(conv3_x, FILTERS[3], KERNEL_SIZE, 1, 'x'+str(frameno)+'_4a', identity_connection=False) # [batch_size, 56, 12, 256]
        for r1 in range(1, 23):
            conv4_x = conv_res_block(conv4_x, FILTERS[3], KERNEL_SIZE, 1, 'x'+str(frameno)+'_4b'+str(r1))
        print("BLOCK3 - X" + str(frameno)+": ", conv4_x.shape)

        # residual block4
        conv5_x = conv_res_block(conv4_x, FILTERS[4], KERNEL_SIZE, 2, 'x'+str(frameno)+'_5a', identity_connection=False) # [batch_size, 28, 6, 512]
        conv5_x = conv_res_block(conv5_x, FILTERS[4], KERNEL_SIZE, 1, 'x'+str(frameno)+'_5b')
        conv5_x = conv_res_block(conv5_x, FILTERS[4], KERNEL_SIZE, 1, 'x'+str(frameno)+'_5c')
        print("BLOCK4 - X" + str(frameno)+": ", conv5_x.shape)

        return conv1_x, conv2_x, conv3_x, conv5_x
        #return convlstm1_o, convlstm1_s, convlstm2_o, convlstm2_s, convlstm3_o, convlstm3_s, conv5_x

def encoder_lstm(y, feat_shape, mode):
    training = tf.equal(mode, 'train')
    convlstm_o, convlstm_s = convlstm_block(feat_shape, y, 1, None, 'encoder_convlstm_') # [batch_size, 28, 6, 512]
    print('convlstm_s: ', convlstm_s)
    return convlstm_o, convlstm_s

def decoder_net(skip1, skip2, skip3, skip4):
    with tf.device('/gpu:0'):
        learned_output = skip4
        print('learned_output: ', learned_output.shape)
        conv_feat = conv2d_layer(skip4, RES_FILTERS[4], 1, 1, PADDING_SAME) # [batch_size, 28, 6, 512]
        print('conv_feat: ', conv_feat.shape)

        up1 = tf.layers.conv2d_transpose(inputs=conv_feat, filters=RES_FILTERS[2], kernel_size=[3, 3], padding="same", strides=2, activation=tf.nn.relu) # [batch_size, 56, 12, 128]
        print('up1: ', up1.shape)
        skip_1 = tf.add(up1, skip3, name="skip_1")
        #print('skip_1: ', skip_1.shape)
        conv6 = conv2d_layer(skip_1, RES_FILTERS[2], 1, 1, PADDING_SAME)
        #print('conv6: ', conv6.shape)

        up2 = tf.layers.conv2d_transpose(inputs=conv6, filters=RES_FILTERS[1], kernel_size=[2, 3], padding="valid", strides=2, activation=tf.nn.relu) # [batch_size, 112, 25, 64]
        print('up2: ', up2.shape)
        skip_2 = tf.add(up2, skip2, name="skip_2")
        #print('skip_2: ', skip_2.shape)
        conv7 = conv2d_layer(skip_2, RES_FILTERS[1], 1, 1, PADDING_SAME)
        #print('conv7: ', conv7.shape)

        up3 = tf.layers.conv2d_transpose(inputs=conv7, filters=FILTERS[1], kernel_size=[3, 2], padding="valid", strides=2, activation=tf.nn.relu) # [batch_size, 225, 50, 16]
        print('up3: ', up3.shape)
        skip_3 = tf.add(up3, skip1, name="skip_3")
        #print('skip_3: ', skip_3.shape)
        conv8 = conv2d_layer(skip_3, FILTERS[1], 1, 1, PADDING_SAME)
        #print('conv8: ', conv8.shape)

        up4 = tf.layers.conv2d_transpose(inputs=conv8, filters=FILTERS[1], kernel_size=[3, 3], padding="same", strides=2, activation=tf.nn.relu) # [batch_size, 450, 100, 16]
        print('up4: ', up4.shape)

        logits = tf.layers.conv2d(inputs=up4, filters=1, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=(1, 1), padding='same', activation=tf.nn.relu) # [batch_size, 450, 100, 1]
        print('logits: ', logits.shape)

        pred = tf.layers.conv2d(inputs=logits, filters=1, kernel_size=[1, 1], strides=(1, 1), padding='same', activation=tf.nn.sigmoid) # [batch_size, 450, 100, 1]
        print('predictions: ', pred.shape)
        return logits, pred

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

def iou_metric_1(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, dtype=tf.bool)
    y_pred = tf.cast(y_pred, dtype=tf.bool)
    intersection = tf.cast(tf.logical_and(y_true, y_pred), dtype=tf.int32)
    union = tf.cast(tf.logical_or(y_true, y_pred), dtype=tf.int32)
    iou = tf.reduce_sum(intersection)/tf.reduce_sum(union) + smooth
    return iou

def iou_metric_2(tp, fp, fn):
    smooth = 1e-6
    return tf.reduce_sum(tp) / tf.reduce_sum(tp) + tf.reduce_sum(fp) + tf.reduce_sum(fn) + smooth

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    loss = 1 - score
    return loss

def weighted_bce_loss(y_true, y_pred):
    loss = BETA * (y_true * (-tf.log(y_pred + eps))) + (1 - BETA) * ((1 - y_true) * (-tf.log((1 - y_pred) + eps)))
    return loss

def get_confusion_matrix(y_true, y_pred):
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

def compute_threshold(frame, th):
    threshold = th
    frame[frame > th] = 1
    frame[frame <= th] = 0
    return frame

def build_model(input_stack, mode):
    print('Encoder - Downsampling ........')
    encoder_output = []
    #convlstm1_s = None
    #convlstm2_s = None
    #convlstm3_s = None
    for inp in range(0, len(input_stack)):
        conv1_x, conv2_x, conv3_x, enc_output = encoder_net(input_stack[inp], inp+1, mode)
        encoder_output.append(enc_output)

    print('Encoder - ConvLSTM ........')
    feat_shape = encoder_output[-1].get_shape().as_list()[1:]
    enc_convlstm_o, enc_convlstm_s = encoder_lstm(encoder_output, feat_shape, mode)

    print('Decoder - Upsampling ........')
    logits, pred = decoder_net(conv1_x, conv2_x, conv3_x, enc_convlstm_o)
    return logits, pred

def main(unused_argv):
    TRAIN_FEATURES = []
    TRAIN_LABELS = []
    VALIDATION_FEATURES = []
    VALIDATION_LABELS = []
    TEST_FEATURES = []
    TEST_LABELS = []
    gpu_fraction = 0.9
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth = True), allow_soft_placement=True)
    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.device('/gpu:0'):
        # tf Graph input
        x1_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        x2_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        x3_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH, 1])
        y_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGTH, INPUT_WIDTH])
        lr_placeholder = tf.placeholder(tf.float32)
        mode = tf.placeholder(tf.string, name='mode')

        input_stack = [x1_placeholder, x2_placeholder, x3_placeholder]
        logits, pred = build_model(input_stack, mode)
        logits = tf.reshape(logits, [BATCH_SIZE, -1],  name="logits_tensor")
        print('logits: ', logits)
        pred = tf.reshape(pred, [BATCH_SIZE, -1], name="sigmoid_tensor")
        print('pred: ', pred)

        if mode != 'test':
            labels = tf.reshape(y_placeholder, [BATCH_SIZE, 45000])
            #loss = ALPHA * weighted_bce_loss(labels, pred) + (1 - ALPHA) * dice_loss(labels, pred)
            loss = weighted_bce_loss(labels, pred)
            loss = tf.reduce_mean(loss, name='loss')
            print('loss: ', loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_placeholder)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            # training & evaluation metrics
            ## Round our sigmoid [0,1] output to be either 0 or 1
            pred_integer = tf.cast(tf.round(pred), tf.int32, name="Prediction_Integer")
            #pred_integer = tf.cast(tf.greater(pred, 0.5), dtype=tf.int32)
            labels_integer = tf.cast(tf.round(labels), tf.int32, name="labels_Integer")
            ## Define operations for {true, false} {positive, negative} predictions
            TP, TN, FP, FN = get_confusion_matrix(labels_integer, pred_integer)
            print('TP, TN, FP, FN: ', TP, TN, FP, FN)
            accuracy_op, precision_op, recall_op, f1 = get_accuracy_recall_precision_f1score(TP, TN, FP, FN)
            print('accuracy_op, precision_op, recall_op, f1: ', accuracy_op, precision_op, recall_op, f1)

            # intersection over union metric.
            iou1_ = iou_metric_1(labels_integer, pred_integer)
            #print('iou1: ', iou1_.shape)
            miou1_ = tf.reduce_mean(iou1_)
            #print('miou1: ', miou1_.shape)
            iou2_ = iou_metric_2(TP, FP, FN)
            #print('iou2: ', iou2_.shape)
            miou2_ = tf.reduce_mean(iou2_)
            #print('miou2: ', miou2_.shape)

            # F1 score/Dice coeff metric
            dice_score1_ = dice_coeff(labels_integer, pred_integer)
            #print('dice_score1: ', dice_score1_)
            dice_score2_ = original_dice_coef(TP, FP, FN)
            #print('dice_score2: ', dice_score2_)

            #smooth = 1e-6
            #intersection = tf.reduce_sum(labels_integer * pred_integer)
            #score = ((2.0 * intersection) + smooth) / (tf.reduce_sum(labels_integer) + tf.reduce_sum(pred_integer) + smooth)
            #score = (2. * intersection)/(tf.reduce_sum(labels_integer) + tf.reduce_sum(pred_integer) + 1e-4)
            #print('dice_coeff: ', score)

            is_correct = tf.equal(pred_integer, labels_integer)
            pixel_accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='pixel_accuracy')
            mean_class_accuracy, mean_class_acc_op = tf.metrics.mean_per_class_accuracy(labels_integer, pred_integer, 2, name='mean_class_accuracy')
            mean_iou, mean_op = tf.metrics.mean_iou(labels_integer, pred_integer, 2, name='mean_iou')
            auc, auc_op = tf.metrics.auc(labels_integer, pred, curve='ROC', name='auc_roc')

            #accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions=pred)
            #precision, precision_op = tf.metrics.precision(labels=labels, predictions=pred)
            #recall, recall_op = tf.metrics.recall(labels=labels, predictions=pred)

            with tf.name_scope('net_model_summary'):
                summary_loss = tf.summary.scalar('loss', loss)
                summary_pre = tf.summary.scalar('precision', precision_op)
                summary_rec = tf.summary.scalar('recall', recall_op)
                summary_miou = tf.summary.scalar('mean_iou', mean_iou)

                #summary_miou1 = tf.summary.scalar('mean_iou1', miou1_)
                #summary_miou2 = tf.summary.scalar('mean_iou2', miou2_)

                summary_dice1 = tf.summary.scalar('dice_score1_', dice_score1_)
                #summary_dice2 = tf.summary.scalar('dice_score2_', dice_score2_)

                summary_class_acc = tf.summary.scalar('mean_class_accuracy', mean_class_accuracy)
                #summary_acc = tf.summary.scalar('accuracy', accuracy_op)
                #summary_pre = tf.summary.scalar('precision', precision_op)
                #summary_rec = tf.summary.scalar('recall', recall_op)
                summary_pixacc = tf.summary.scalar('pixel_accuracy', pixel_accuracy)
                summary_acc = tf.summary.scalar('accuracy', accuracy_op)
                summary_auc = tf.summary.scalar('auc_roc', auc)
                #summary_op = tf.summary.merge([summary_loss, summary_pixacc, summary_class_acc, summary_miou, summary_acc, summary_pre, summary_rec], name='network_summary')
                summary_op = tf.summary.merge([summary_loss, summary_pre, summary_rec, summary_miou, summary_dice1, summary_acc, summary_auc], name='net_model_summary')

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
                total_accuracy_value = 0
                total_precision_value = 0
                total_recall_value = 0
                total_miou_value = 0
                total_pix_acc_value = 0
                total_mean_class_acc_value = 0
                total_auc = 0

                total_dice1 = 0
                total_dice2 = 0
                total_iou1 = 0
                total_iou2 = 0
                total_miou1 = 0
                total_miou2 = 0

                TRAIN_SET, VALIDATION_SET, TEST_SET = read_samples()
                iteration_no = int(len(TRAIN_SET)/BATCH_SIZE)
                print("Epoch: {}/{} - ".format(epoch+1, int(EPOCHS)))
                for batchno in range(0, int(len(TRAIN_SET)/BATCH_SIZE)):
                    t_set = []
                    for index in range(0, BATCH_SIZE):
                        t_set.append(read_sample_data(TRAIN_SET[(batchno * BATCH_SIZE) + index]))
                    for x in range(len(t_set)):
                        data_row = t_set[x][0]
                        TRAIN_FEATURES.append([data_row[0].todense(), data_row[1].todense(), data_row[2].todense()])
                        TRAIN_LABELS.append(data_row[3].todense())
                    train_data = np.asarray(TRAIN_FEATURES, dtype=np.float32)
                    train_labels = np.asarray(TRAIN_LABELS, dtype=np.float32)
                    x1, x2, x3 = reshape_input(train_data)
                    feed = {x1_placeholder: x1, x2_placeholder: x2, x3_placeholder: x3, y_placeholder: train_labels, lr_placeholder: LEARNING_RATE, mode:'train'}
                    _, loss_value, pix_acc, class_acc, miou, acc, pre, rec, dice_score1, auc_roc, summary_value = sess.run([train_op, loss, pixel_accuracy, mean_class_accuracy,
                    mean_iou, accuracy_op, precision_op, recall_op, dice_score1_, auc, summary_op], feed_dict = feed)

                    #iou1, miou1, iou2, miou2, dice_score1, dice_score2 = sess.run([iou1_, miou1_, iou2_, miou2_, dice_score1_, dice_score2_], feed_dict = feed)
                    #sess.run([mean_class_acc_op, mean_op, accuracy_op, precision_op, recall_op], feed_dict = feed)
                    *_, mop, aucop, tp, fp, fn = sess.run([mean_class_acc_op, mean_op, auc_op, TP, FP, FN], feed_dict = feed)
                    total_loss_value += loss_value
                    total_accuracy_value += acc
                    total_precision_value += pre
                    total_recall_value += rec
                    total_miou_value += miou
                    total_mean_class_acc_value += class_acc
                    total_pix_acc_value += pix_acc
                    total_auc += auc_roc

                    total_dice1 += dice_score1
                    '''
                    total_dice2 += dice_score2
                    total_iou1 += iou1
                    total_iou2 += iou2
                    total_miou1 += miou1
                    total_miou2 += miou2
                    '''

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
                avg_accuracy = total_accuracy_value/iteration_no
                avg_precision = total_precision_value/iteration_no
                avg_recall = total_recall_value/iteration_no
                avg_miou = total_miou_value/iteration_no
                avg_mean_class_acc = total_mean_class_acc_value/iteration_no
                avg_pix_acc = total_pix_acc_value/iteration_no
                avg_auc = total_auc/iteration_no

                avg_dice1 = total_dice1/iteration_no
                '''
                avg_dice2 = total_dice2/iteration_no
                avg_iou1 = total_iou1/iteration_no
                avg_iou2 = total_iou2/iteration_no
                avg_miou1 = total_miou1/iteration_no
                avg_miou2 = total_miou2/iteration_no
                '''

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
                total_validation_acc_value = 0
                total_validation_pre_value = 0
                total_validation_rec_value = 0
                total_validation_miou_value = 0
                total_validation_pix_acc_value = 0
                total_validation_mean_class_acc_value = 0
                total_auc_v = 0

                total_validation_dice1 = 0
                total_validation_dice2 = 0
                total_validation_iou1 = 0
                total_validation_iou2 = 0
                total_validation_miou1 = 0
                total_validation_miou2 = 0

                for valno in range(0, int(len(VALIDATION_SET)/BATCH_SIZE)):
                    val_set = []
                    for idx in range(0, BATCH_SIZE):
                        val_set.append(read_sample_data(VALIDATION_SET[(valno * BATCH_SIZE) + idx]))
                    VALIDATION_FEATURES = []
                    VALIDATION_LABELS = []
                    for y in range(len(val_set)):
                        data_row = val_set[y][0]
                        VALIDATION_FEATURES.append([data_row[0].todense(), data_row[1].todense(), data_row[2].todense()])
                        VALIDATION_LABELS.append(data_row[3].todense())

                    eval_data = np.asarray(VALIDATION_FEATURES, dtype=np.float32)
                    eval_labels = np.asarray(VALIDATION_LABELS, dtype=np.float32)
                    x1, x2, x3 = reshape_input(eval_data)
                    feed = {x1_placeholder: x1, x2_placeholder: x2, x3_placeholder: x3, y_placeholder: eval_labels, lr_placeholder: LEARNING_RATE, mode:'validation'}
                    valid_loss, pix_acc, class_acc, miou, valid_acc, pre, rec, dice_score1, auc_roc_v, summary_value_valid = sess.run([loss, pixel_accuracy, mean_class_accuracy,
                    mean_iou, accuracy_op, precision_op, recall_op, dice_score1_, auc, summary_op], feed_dict= feed)

                    #iou1, miou1, iou2, miou2, dice_score1, dice_score2 = sess.run([iou1_, miou1_, iou2_, miou2_, dice_score1_, dice_score2_], feed_dict = feed)
                    *_, val_mop, aucop_v, val_tp, val_fp, val_fn = sess.run([mean_class_acc_op, mean_op, auc_op, TP, FP, FN], feed_dict = feed)
                    #sess.run([mean_class_acc_op, mean_op, accuracy_op, precision_op, recall_op], feed_dict = feed)

                    total_validation_loss_value += valid_loss
                    total_validation_acc_value += valid_acc
                    total_validation_pre_value += pre
                    total_validation_rec_value += rec
                    total_validation_miou_value += miou
                    total_validation_pix_acc_value += pix_acc
                    total_validation_mean_class_acc_value += class_acc
                    total_auc_v += auc_roc_v

                    total_validation_dice1 += dice_score1
                    '''
                    total_validation_dice2 += dice_score2
                    total_validation_iou1 += iou1
                    total_validation_iou2 += iou2
                    total_validation_miou1 += miou1
                    total_validation_miou2 += miou2
                    '''

                    validation_writer.add_summary(summary_value_valid, epoch * itr_validation + valno)

                    #if (valno+1) % 25 == 0:
                        #print("     Iteration: {}/{}, Loss: {:.8f}, Accuracy: {:.8f}, Precision: {:.8f}, Recall: {:.8f}, PixelAccuracy: {:.8f}, MeanClassAccuracy: {:.3f}, Mean_IoU: {:.3f}".format(valno+1,
                        #itr_validation, total_validation_loss_value, total_validation_acc_value, pre, rec, pix_acc, class_acc, miou))
                        #save_path = saver.save(sess, MODEL_DIR + "model.ckpt")
                        #saver.save(sess, MODEL_DIR + "model.ckpt")

                    VALIDATION_FEATURES = []
                    VALIDATION_LABELS = []

                avg_validation_loss = total_validation_loss_value/itr_validation
                avg_validation_acc = total_validation_acc_value/itr_validation
                avg_validation_pre = total_validation_pre_value/itr_validation
                avg_validation_rec = total_validation_rec_value/itr_validation
                avg_validation_miou = total_validation_miou_value/itr_validation
                avg_validation_mean_class_acc = total_validation_mean_class_acc_value/itr_validation
                avg_validation_pix_acc = total_validation_pix_acc_value/itr_validation
                avg_total_auc_v = total_auc_v/itr_validation

                avg_validation_dice1 = total_validation_dice1/itr_validation
                '''
                avg_validation_dice2 = total_validation_dice2/itr_validation
                avg_validation_iou1 = total_validation_iou1/itr_validation
                avg_validation_iou2 = total_validation_iou2/itr_validation
                avg_validation_miou1 = total_validation_miou1/itr_validation
                avg_validation_miou2 = total_validation_miou2/itr_validation
                '''

                end = time.time()
                duration = end - start
                #print("     Validation Iteration: {}/{} | {:.2f}s  - Loss: {:.8f}%, Accuracy: {:.8f}%, Precision: {:.8f}%, Recall: {:.8f}%, PixelAccuracy: {:.8f}%, MeanClassAccuracy: {:.3f}%, Mean_IoU: {:.3f}%".format(itr_validation,
                #itr_validation, duration, avg_validation_loss * 100, avg_validation_acc * 100, avg_validation_pre * 100, avg_validation_rec * 100, avg_validation_pix_acc * 100, avg_validation_mean_class_acc * 100, avg_validation_miou * 100))
                #if avg_validation_acc > avg_accuracy:
                print("[{}/{}] | {:.2f}s - ".format(iteration_no, iteration_no, duration))
                #print("     Train: loss: {:.6f}%, pre: {:.6f}%, rec: {:.6f}%, m_iou: {:.4f}%, iou1: {:.4f}%, m_iou1: {:.4f}%, iou2: {:.4f}%, m_iou2: {:.4f}%, dice1: {:.4f}%, dice2: {:.4f}%, m_acc: {:.3f}%, pix_acc: {:.4f}%, acc: {:.4f}%".format(avg_loss * 100,
                #avg_precision * 100, avg_recall * 100, avg_miou * 100, avg_iou1 * 100, avg_miou1 * 100, avg_iou2 * 100, avg_miou2 * 100,  avg_dice1 * 100, avg_dice2 * 100, avg_mean_class_acc * 100, avg_pix_acc * 100, avg_accuracy * 100))
                print("     Train: loss: {:.6f}%, pre: {:.6f}%, rec: {:.6f}%, m_iou: {:.4f}%, dice1: {:.4f}%, m_acc: {:.3f}%, pix_acc: {:.4f}%, acc: {:.4f}%, auc: {:.4f}".format(avg_loss * 100,
                avg_precision * 100, avg_recall * 100, avg_miou * 100, avg_dice1 * 100, avg_mean_class_acc * 100, avg_pix_acc * 100, avg_accuracy * 100, avg_auc * 100))

                print("[{}/{}] |".format(itr_validation, itr_validation))
                #print("     Valid: loss: {:.6f}%, pre: {:.6f}%, rec: {:.6f}%, m_iou: {:.4f}%, iou1: {:.4f}%, m_iou1: {:.4f}%, iou2: {:.4f}%, m_iou2: {:.4f}%, dice1: {:.4f}%, dice2: {:.4f}%, m_acc: {:.3f}%, pix_acc: {:.4f}%, acc: {:.4f}%".format(avg_validation_loss * 100,
                #avg_validation_pre * 100, avg_validation_rec * 100, avg_validation_miou * 100, avg_validation_iou1 * 100, avg_validation_miou1 * 100, avg_validation_iou2 * 100, avg_validation_miou2 * 100,  avg_validation_dice1 * 100, avg_validation_dice2 * 100, avg_validation_mean_class_acc * 100, avg_validation_pix_acc * 100, avg_validation_acc * 100))
                print("     Valid: loss: {:.6f}%, pre: {:.6f}%, rec: {:.6f}%, m_iou: {:.4f}%, dice1: {:.4f}%, m_acc: {:.3f}%, pix_acc: {:.4f}%, acc: {:.4f}%, auc: {:.4f}".format(avg_validation_loss * 100,
                avg_validation_pre * 100, avg_validation_rec * 100, avg_validation_miou * 100, avg_validation_dice1 * 100, avg_validation_mean_class_acc * 100, avg_validation_pix_acc * 100, avg_validation_acc * 100, avg_total_auc_v * 100))
                print("     Train: tp: {:.2f}, fp: {:.2f}, fn: {:.2f} ".format(tp, fp, fn))
                print("     Valid: tp: {:.2f}, fp: {:.2f}, fn: {:.2f} ".format(val_tp, val_fp, val_fn))

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
                    #if non_update_counter > 2:
                        #lrate = lrate/10

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
                    TEST_FEATURES.append([data_row[0].todense(), data_row[1].todense(), data_row[2].todense()])
                    TEST_LABELS.append(data_row[3].todense())
                test_data = np.asarray(TEST_FEATURES, dtype=np.float32)
                test_y_true = np.asarray(TEST_LABELS, dtype=np.float32)
                x1, x2, x3 = reshape_input(test_data)
                feed = {x1_placeholder: x1, x2_placeholder: x2, x3_placeholder: x3, mode:'test'}
                #feed = {x1_placeholder: x1, x2_placeholder: x2, x3_placeholder: x3, y_placeholder: test_y_true, mode:'test'}

                saver.restore(sess, MODEL_DIR + "model.ckpt")
                print("Model restored from file: %s" % save_path)
                pred_op = sess.run([pred], feed_dict= feed)
                #print('pred_op: ', pred_op.shape)
                #pred_op, pix_acc, class_acc, miou, acc, summary_value_test = sess.run([pred, pixel_accuracy, mean_class_accuracy, mean_iou, accuracy_op, summary_op], feed_dict= feed)
                #print("Testing => Batch: {}/{}, Accuracy: {:.8f}%, PixelAccuracy: {:.4f}%, MeanClassAccuracy: {:.3f}%, Mean_IoU: {:.3f}%".format(t+1,
                #itr_testing_no, acc * 100, pix_acc * 100, class_acc * 100, miou * 100))

                TEST_FEATURES = []
                TEST_LABELS = []
                res = []
                b = 0
                #print(pred_op[0])
                threshold_value = 0.45
                for i, result in enumerate(pred_op[0]):
                    fig = plt.figure(figsize=(14, 14))
                    fig.canvas.set_window_title('Predicted label for target label - ' + str(i+1))
                    fig.add_subplot(2, 1, 1)
                    plt.xlabel("target label grid",fontsize="14")
                    plt.matshow(test_y_true[i].T, fignum=False)

                    label_frame = test_y_true[i]
                    cnt_1 = label_frame[np.where(label_frame == 1.0)]
                    print('label (No. of cars): ', len(cnt_1))
                    frame1 = np.reshape(result, (450, 100))
                    frame2 = np.reshape(result, (450, 100))
                    sorted_frame = sorted(frame2[np.nonzero(frame2)])
                    print('pred label for cars: ', sorted_frame[-(len(cnt_1)+10):])
                    print('pred label> 1:', np.count_nonzero(frame1))
                    print('pred label> 0:', np.count_nonzero(frame1==0))
                    frame1 = compute_threshold(frame1, threshold_value)
                    fig.add_subplot(2, 1, 2)
                    plt.xlabel("predicted label grid",fontsize="14")
                    plt.matshow(frame1.T, fignum=False)
                    cnt_1 = frame1[np.where(frame1 == 1.0)]
                    print('pred - 1: ', cnt_1)
                    print('pred 1 count: ', len(cnt_1))

                    #misc.imsave(os.path.join(MODEL_DIR + 'test/' , 'pred_image_'+ str(t) + "_" + str(i)+ ".png"), frame1)
                    fig.savefig(os.path.join(MODEL_DIR + 'test/' , 'pred_image_'+ str(t) + "_" + str(i)+ ".png"))
                    print('saving image: ' + 'pred_image_'+ str(t) + "_" + str(i)+ ".png")
                    plt.show()

if __name__ == "__main__":
  tf.app.run()
