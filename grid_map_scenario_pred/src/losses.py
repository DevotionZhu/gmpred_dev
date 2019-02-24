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

class network_losses():
    def __init__(self):
        self.alpha = 0.5
        self.beta = 0.6
        self.epsilon = 1e-10
        self.smooth = 1e-6

    def dice_coeff(self, y_true_f, y_pred_f):
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + self.smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - dice_coeff(y_true, y_pred)
        return loss

    def original_dice_loss(self, tp, tn, fp, fn):
        return (2 * tf.reduce_sum(tp) / (tf.reduce_sum(2*tp) + tf.reduce_sum(fp) + tf.reduce_sum(fn) + self.smooth))

    def weighted_bce_loss(self, y_true, y_pred):
        loss = self.beta * (y_true * (-tf.log(y_pred + self.epsilon))) + (1 - self.beta) * ((1 - y_true) * (-tf.log((1 - y_pred) + self.epsilon)))
        return loss

    def get_confusion_matrix(self, y_pred, y_true):
        y_pred = tf.cast(tf.round(y_pred), tf.int32)
        y_true = tf.cast(tf.round(y_true), tf.int32)
        TP = tf.cast(tf.count_nonzero(y_pred * y_true, name='True_Positives', dtype=tf.int32), dtype=tf.float32)
        TN = tf.cast(tf.count_nonzero((y_pred - 1) * (y_true - 1), name="True_Negatives", dtype=tf.int32), dtype=tf.float32)
        FP = tf.cast(tf.count_nonzero(y_pred * (y_true - 1), name="False_Positives", dtype=tf.int32), dtype=tf.float32)
        FN = tf.cast(tf.count_nonzero((y_pred - 1) * y_true, name="False_Negatives", dtype=tf.int32), dtype=tf.float32)

        return TP, TN, FP, FN

    def get_accuracy_recall_precision_f1score(self, tp, tn, fp, fn):
        # accuracy ::= (TP + TN) / (TN + FN + TP + FP)
        accuracy = tf.cast(tf.divide(tp + tn, tp + tn + fp + fn, name="Accuracy"), dtype=tf.float32)
        # precision ::= TP / (TP + FP)
        precision = tf.cast(tf.divide(tp, tp + fp, name="Precision"), dtype=tf.float32)
        # recall ::= TP / (TP + FN)
        recall = tf.cast(tf.divide(tp, tp + fn, name="Recall"), dtype=tf.float32)
        # F1 score ::= 2 * precision * recall / (precision + recall)
        f1 = tf.cast(tf.divide((2 * precision * recall), (precision + recall), name="F1_score"), dtype=tf.float32)
        return accuracy, precision, recall, f1

    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2):
        """
        y : binary class {+1, -1}
        p : probability of input correctly classified to binary class

        Given Cross Entropy (CE) loss for binary classification:
        CE(p, y) =
        -log(p) ,  if y = 1
        -log(1 - p), if y = -1

        The paper introduces the Focal Loss (FL) term as follows
        FL(p,y) =
        -(1-p)^gamma * log(p), if y = +1
        -(p)^gamma * log(1-p), if y = -1

        In practice the authors use an a-balanced variance of FL:
        FL(p,y) =
        -a(y) * ( 1 - p ) ^ gamma * log(p), if y = +1
        -a(y) * ( p )  ^ gamma * log(1 - p), if y = -1
        Where a(y) is a multiplier term fixing the class imbalance. This form yields slightly improved accuracy over the non-a-balanced form.

        Computer focal loss for binary classification
        Args:
          labels: A int32 tensor of shape [batch_size].
          logits: A float32 tensor of shape [batch_size].
          alpha: A scalar for focal loss alpha hyper-parameter.
          gamma: A scalar for focal loss gamma hyper-parameter.
        Returns:
          A tensor of the same shape as `lables`
        """
        focal_loss = -y_true * (1-alpha) * ((1-y_pred) * gamma) * tf.log(y_pred + self.epsilon) - (1 - y_true) * alpha * (y_pred**gamma) * tf.log((1-y_pred) + self.epsilon)
        return focal_loss

    def tversky_loss(self, lab, pred, alpha=0.5, beta=0.5):
        EPSILON = 0.00001
        TP = tf.cast(tf.reduce_sum(lab * pred), dtype=tf.float32)
        TN = tf.cast(tf.reduce_sum((pred - 1) * (lab - 1)), dtype=tf.float32)
        FP = tf.cast(tf.reduce_sum(pred * (lab - 1)), dtype=tf.float32)
        FN = tf.cast(tf.reduce_sum((pred - 1) * lab), dtype=tf.float32)

        fp = alpha * FP
        fn = beta * FN
        numerator = TP
        denominator = TP + fp + fn + self.smooth
        score = numerator / denominator
        return 1.0 - tf.reduce_mean(score)

    '''
    def tversky(self, prediction, ground_truth, weight_map=None, alpha=0.5, beta=0.5):
        """
         Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            alpha: controls the penalty for false positives.
            beta: controls the penalty for false negatives.
            eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
        Function to calculate the Tversky loss for imbalanced data

            Sadegh et al. (2017)

            Tversky loss function for image segmentation
            using 3D fully convolutional deep networks

        :param prediction: the logits
        :param ground_truth: the segmentation ground_truth
        :param alpha: weight of false positives
        :param beta: weight of false negatives
        :param weight_map:
        :return: the loss
        """
        prediction = tf.to_float(prediction)
        if len(ground_truth.shape) == len(prediction.shape):
            ground_truth = ground_truth[..., -1]
        one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
        one_hot = tf.sparse_tensor_to_dense(one_hot)

        p0 = prediction
        p1 = 1 - prediction
        g0 = one_hot
        g1 = 1 - one_hot

        if weight_map is not None:
            num_classes = prediction.shape[1].value
            weight_map_flattened = tf.reshape(weight_map, [-1])
            weight_map_expanded = tf.expand_dims(weight_map_flattened, 1)
            weight_map_nclasses = tf.tile(weight_map_expanded, [1, num_classes])
        else:
            weight_map_nclasses = 1

        tp = tf.reduce_sum(weight_map_nclasses * p0 * g0)
        fp = alpha * tf.reduce_sum(weight_map_nclasses * p0 * g1)
        fn = beta * tf.reduce_sum(weight_map_nclasses * p1 * g0)

        EPSILON = 0.00001
        numerator = tp
        denominator = tp + fp + fn + EPSILON
        score = numerator / denominator
        return 1.0 - tf.reduce_mean(score)
        '''
