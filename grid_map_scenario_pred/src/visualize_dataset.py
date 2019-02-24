import numpy as np
import math
#import cv2
import scipy.misc
import scipy.sparse
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
from os import listdir, makedirs
from os.path import isfile, join, dirname, exists, basename
import shutil
import itertools
import xml.etree.ElementTree as ET
import pickle
#from PIL import Image, ImageDraw
import colorsys #hsv to rgb transformation for radar colors
from collections import OrderedDict
from configuration import Configuration
import time
import glob
import random

np.set_printoptions(threshold=np.nan)


class VisualizeDataset():
    def __init__(self):
        self.conf = Configuration()
        self.dataset_path = self.conf.DATASET_SAVING_PATH + 'ds/'

        '''
        # 1 time step ahead frame prediction
        self.DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_4frames_1ts/"
        self.TOTAL_SAMPLES = 4776
        self.TRAIN_SAMPLES = 4000
        self.VALIDATION_SAMPLES = 500
        self.TEST_SAMPLES = 276
        '''
        '''
        # 2 time step ahead frame prediction
        self.DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_4frames_2ts/"
        self.TOTAL_SAMPLES = 4740
        self.TRAIN_SAMPLES = 4000
        self.VALIDATION_SAMPLES = 500
        self.TEST_SAMPLES = 240
        '''
        '''
        # 3 time step ahead frame prediction
        self.DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_4frames_3ts/"
        self.TOTAL_SAMPLES = 4704
        self.TRAIN_SAMPLES = 4000
        self.VALIDATION_SAMPLES = 500
        self.TEST_SAMPLES = 204
        '''

        # 1 time step ahead frame prediction
        self.DATASET_DIR = "/home/mushfiqrahman/dataset/db_numpy/samples_10frames_1ts/"
        self.TOTAL_SAMPLES = 4704
        self.TRAIN_SAMPLES = 4000
        self.VALIDATION_SAMPLES = 600
        self.TEST_SAMPLES = 104

        self.features_hist_list = []
        self.labels_hist_list = []
        #pygame.init()

        self.TRAIN_SET = []
        self.VALIDATION_SET = []
        self.TEST_SET = []

    # Reading compressed npz file.
    def read_dataset(self):
        print('--- Reading Dataset -> dataset.npz file ---')
        compressed_dataset = np.load(self.dataset_path + 'dataset_' + str(self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR) + '.npz')
        print(compressed_dataset.files)
        print('after loading, 1st row:>', compressed_dataset['arr_0'][0])
        print(str(len(compressed_dataset['arr_0'])))

        data_row = compressed_dataset['arr_0'][0] # taking first row the dataset [feat1, feat2, feat3, label1]
        print('first arr>', data_row[0]) # test print of first element of first row.
        #print(sparse.issparse(data_row[0]))

    def visualize_features_labels_by_histogram(self):
        N = self.conf.NUMBER_OF_IMAGES_ARRAY
        for x in range(0, N-3):
            for i in range(x+1, N-2):
                if i - x == self.conf.SLIDING_FEATURE_BY_FRAME_WINDOW_FACTOR:
                    for j in range(i+1, N-1):
                        if j - i == self.conf.SLIDING_FEATURE_BY_FRAME_WINDOW_FACTOR:
                            for k in range(j+1, N):
                                if k - j <= self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR:
                                    feat1 = self.conf.SORTED_NP_FILES[x]
                                    feat2 = self.conf.SORTED_NP_FILES[i]
                                    feat3 = self.conf.SORTED_NP_FILES[j]
                                    label1 = self.conf.SORTED_NP_FILES[k]
                                    diff1 = int(round(feat2 - feat1))
                                    self.features_hist_list.append(diff1)

                                    diff2 = int(round(feat3 - feat2))
                                    self.features_hist_list.append(diff2)
                                    #print('diff1- ' + str(diff1) + ' diff2- ' + str(diff2))

                                    diff3 = int(round(label1 - feat3))
                                    self.labels_hist_list.append(diff3)

        #print(len(self.features_hist_list))
        feat_hist_list = np.asarray(self.features_hist_list)
        label_hist_list = np.asarray(self.labels_hist_list)
        feat_unique, feat_unique_counts = np.unique(feat_hist_list, return_counts=True)
        print(feat_unique)
        print(len(feat_unique))

        a = []
        a_c = []
        for i in range(6, len(feat_unique)):
            if i < 45:
                a.append(feat_unique[i])
                a_c.append(feat_unique_counts[i])

        print(a)
        print(a_c)

        feat_unique = []
        feat_unique_counts = []
        feat_unique = a
        feat_unique_counts = a_c



        feat_hist_dict = np.asarray((feat_unique, feat_unique_counts))
        label_unique, label_unique_counts = np.unique(label_hist_list, return_counts=True)
        print(label_unique)


        b = []
        b_c = []
        for j in range(10, len(label_unique)):
            if j < 45:
                b.append(label_unique[j])
                b_c.append(label_unique_counts[j])

        print(b)
        print(b_c)

        label_unique = []
        label_unique_counts = []
        label_unique = b
        label_unique_counts = b_c



        label_hist_dict = np.asarray((label_unique, label_unique_counts))

        # histogram 1 - difference between features value (ms)
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Window - Fequecy Histogram of Features Interval')
        indices = np.arange(len(feat_unique))
        width = 0.85
        #plt.title("Fequecy Histogram of Features Interval")
        plt.rcParams["font.family"] = "DejaVu Sans Mono"
        rects = plt.bar(indices, feat_unique_counts, width, color='r')
        plt.xlabel("Time deltas between detection frames (in milisecond)", fontsize=20)
        plt.ylabel("Count", fontsize=20)
        plt.xticks(indices, feat_unique, rotation='vertical')
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., 1.00*height,
            '%d' % int(height), ha='center', va='bottom')
        #plt.tight_layout()

        fig.savefig(os.path.join(self.conf.DATASET_SAVING_PATH + 'hist/', 'hist_1.png'))
        print('saving image: hist_1.png')
        plt.show()
        print('--- Fequecy Histogram of Features Interval visualized successfully ---')


        # histogram 2 - difference between last feature and label value (ms)
        fig, ax = plt.subplots()
        indices = np.arange(len(label_unique))
        fig.canvas.set_window_title('Window - Fequecy Histogram of Labels and Last Features Interval')
        #plt.title("Fequecy Histogram of Labels and Last Features Interval")
        rects = plt.bar(indices, label_unique_counts, width, color='b')
        plt.rcParams["font.family"] = "DejaVu Sans Mono"
        plt.xlabel("Time deltas between Ground Truth (GT) Grid and Last Feature Grid (in milisecond)", fontsize=20)
        plt.ylabel("Count", fontsize=20)
        plt.xticks(indices, label_unique, rotation='vertical')
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., 1.00*height,
            '%d' % int(height), ha='center', va='bottom')
        #plt.tight_layout()
        fig.savefig(os.path.join(self.conf.DATASET_SAVING_PATH + 'hist/', 'hist_2.png'))
        print('saving image: hist_2.png')
        plt.show()
        print('--- Fequecy Histogram of Labels and Last Features Interval visualized successfully ---')

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05 * height, '%d' % int(height), ha='center', va='bottom')

    def read_samples(self):
        TRAIN_SET = []
        VALIDATION_SET = []
        TEST_SET = []
        TRAIN_VAL_SET = []

        file_list = glob.glob(self.DATASET_DIR + "*.npz")
        sorted_list = sorted(file_list)
        #print('sorted_list {}'.format(sorted_list))

        # divide dataset into 2 sets: i) train+validation set ii) test set
        # and shuffle only set (i) and later divide shuffle set into train+validation sets.
        for sample in range(0, self.TRAIN_SAMPLES + self.VALIDATION_SAMPLES):
            TRAIN_VAL_SET.append(sorted_list[sample])
        #random.shuffle(TRAIN_VAL_SET)

        for i in range(0, self.TRAIN_SAMPLES):
            TRAIN_SET.append(sorted_list[i])
            #TRAIN_SET.append(TRAIN_VAL_SET[i])
        #print('Sorted TRAIN LIST: ', TRAIN_SET)

        for j in range(self.TRAIN_SAMPLES, self.TRAIN_SAMPLES + self.VALIDATION_SAMPLES):
            VALIDATION_SET.append(sorted_list[j])
            #VALIDATION_SET.append(TRAIN_VAL_SET[j])

        for k in range(self.TRAIN_SAMPLES + self.VALIDATION_SAMPLES, self.TRAIN_SAMPLES + self.VALIDATION_SAMPLES + self.TEST_SAMPLES):
            TEST_SET.append(sorted_list[k])

        random.shuffle(self.TRAIN_SET)
        #print('shuffle_train_list: {}'.format(TRAIN_SET))
        return TRAIN_SET, VALIDATION_SET, TEST_SET

    def read_sample_data(self, path):
        with np.load(path) as data:
            #print('sample - ', path)
            return data['arr_0']

    def visualized_1ts_dataset(self):
        self.TRAIN_SET, self.VALIDATION_SET, self.TEST_SET = self.read_samples()
        print('--- Reading & visualizing Dataset -> file ---')
        #compressed_dataset = np.load(self.dataset_path + 'dataset_' + str(self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR) + '.npz')
        print(len(self.TEST_SET))
        for t in range(0, int(len(self.TEST_SET))):
            #print('test batch: ', str(t + 1))
            test_set = []
            test_set.append(self.read_sample_data(self.TEST_SET[t]))

            row = test_set[0][0]

            fig = plt.figure(figsize=(14, 14))
            fig.canvas.set_window_title('Dataset Row - ' + str(t+1))
            fig.add_subplot(1, 4, 1)
            plt.xlabel("Grid-1",fontsize="14")
            plt.matshow(row[0].todense(), fignum=False)

            fig.add_subplot(1, 4, 2)
            plt.xlabel("Grid-2",fontsize="14")
            plt.matshow(row[1].todense(), fignum=False)

            fig.add_subplot(1, 4, 3)
            plt.xlabel("Grid-3",fontsize="14")
            plt.matshow(row[2].todense(), fignum=False)

            fig.add_subplot(1, 4, 4)
            plt.xlabel("GT. Grid",fontsize="14")
            plt.matshow(row[3].todense(), fignum=False)

            fig.savefig(os.path.join(self.conf.DATASET_SAVING_PATH + 'hist/sample_3ts', 'img_row_' + str(t+1) + '.png'))
            plt.show()

        print('--- batch dataset visualized successfully ---')

    def visualized_seq_1ts_dataset(self):
        self.TRAIN_SET, self.VALIDATION_SET, self.TEST_SET = self.read_samples()
        print('--- Reading & visualizing Dataset -> file ---')
        #compressed_dataset = np.load(self.dataset_path + 'dataset_' + str(self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR) + '.npz')
        print(len(self.TEST_SET))
        for t in range(0, int(len(self.TEST_SET))):
            #print('test batch: ', str(t + 1))
            test_set = []
            test_set.append(self.read_sample_data(self.TEST_SET[t]))

            row = test_set[0][0]

            fig = plt.figure(figsize=(14, 14))
            fig.canvas.set_window_title('Dataset Row - ' + str(t+1))
            fig.add_subplot(2, 5, 1)
            plt.xlabel("Grid-1",fontsize="14")
            plt.matshow(row[0].todense(), fignum=False)

            fig.add_subplot(2, 5, 2)
            plt.xlabel("Grid-2",fontsize="14")
            plt.matshow(row[1].todense(), fignum=False)

            fig.add_subplot(2, 5, 3)
            plt.xlabel("Grid-3",fontsize="14")
            plt.matshow(row[2].todense(), fignum=False)

            fig.add_subplot(2, 5, 4)
            plt.xlabel("Grid-4",fontsize="14")
            plt.matshow(row[3].todense(), fignum=False)

            fig.add_subplot(2, 5, 5)
            plt.xlabel("Grid-5",fontsize="14")
            plt.matshow(row[4].todense(), fignum=False)

            fig.add_subplot(2, 5, 6)
            plt.xlabel("GT. Grid-6",fontsize="14")
            plt.matshow(row[5].todense(), fignum=False)

            fig.add_subplot(2, 5, 7)
            plt.xlabel("GT. Grid-7",fontsize="14")
            plt.matshow(row[6].todense(), fignum=False)

            fig.add_subplot(2, 5, 8)
            plt.xlabel("GT. Grid-8",fontsize="14")
            plt.matshow(row[7].todense(), fignum=False)

            fig.add_subplot(2, 5, 9)
            plt.xlabel("GT. Grid-9",fontsize="14")
            plt.matshow(row[8].todense(), fignum=False)

            fig.add_subplot(2, 5, 10)
            plt.xlabel("GT. Grid-10",fontsize="14")
            plt.matshow(row[9].todense(), fignum=False)

            fig.savefig(os.path.join(self.conf.DATASET_SAVING_PATH + 'hist/sample_seq_1ts', 'img_row_' + str(t+1) + '.png'))
            plt.show()

        print('--- batch dataset visualized successfully ---')

    def visualized_seq_dataset(self):
        self.TRAIN_SET, self.VALIDATION_SET, self.TEST_SET = self.read_samples()
        print('--- Reading & visualizing Dataset -> file ---')
        #compressed_dataset = np.load(self.dataset_path + 'dataset_' + str(self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR) + '.npz')
        print(len(self.TEST_SET))
        for t in range(0, int(len(self.TEST_SET))):
            #print('test batch: ', str(t + 1))
            test_set = []
            test_set.append(self.read_sample_data(self.TEST_SET[t]))

            row = test_set[0][0]

            fig = plt.figure(figsize=(14, 14))
            fig.canvas.set_window_title('Dataset Row - ' + str(t+1))
            fig.add_subplot(1, 10, 1)
            plt.xlabel("Grid-1",fontsize="14")
            plt.matshow(row[0].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 2)
            plt.xlabel("Grid-2",fontsize="14")
            plt.matshow(row[1].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 3)
            plt.xlabel("Grid-3",fontsize="14")
            plt.matshow(row[2].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 4)
            plt.xlabel("Grid-4",fontsize="14")
            plt.matshow(row[3].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 5)
            plt.xlabel("Grid-5",fontsize="14")
            plt.matshow(row[4].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 6)
            plt.xlabel("GT. Grid-6",fontsize="14")
            plt.matshow(row[5].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 7)
            plt.xlabel("GT. Grid-7",fontsize="14")
            plt.matshow(row[6].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 8)
            plt.xlabel("GT. Grid-8",fontsize="14")
            plt.matshow(row[7].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 9)
            plt.xlabel("GT. Grid-9",fontsize="14")
            plt.matshow(row[8].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.add_subplot(1, 10, 10)
            plt.xlabel("GT. Grid-10",fontsize="14")
            plt.matshow(row[9].todense(), fignum=False)
            #plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            fig.savefig(os.path.join(self.conf.DATASET_SAVING_PATH + 'hist/sample_seq_1ts_1', 'img_row_' + str(t+1) + '.png'))
            plt.show()

        print('--- batch dataset visualized successfully ---')

    def visualize_batch_dataset(self):
        #self.read_dataset()
        print('--- Reading & visualizing Dataset -> dataset.npz file ---')
        #data_array = np.load(self.conf.IMAGE_ARRAY_PATH + '225.427038.npy')
        compressed_dataset = np.load(self.dataset_path + 'dataset_' + str(self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR) + '.npz')
        for row in range(0, 4887):
            row = compressed_dataset['arr_0'][row]
            #fig, ax = plt.subplots()
            #fig.canvas.set_window_title('Dataset Row')
            for i in range(0,4):
                plt.matshow(row[i].todense().T)
                plt.show()
                #if i==3:
                    #fig.canvas.set_window_title('Label Frame')
                    #plt.title('Label')
                #else:
                    #fig.canvas.set_window_title('Feature Frame')
                    #plt.title('Feature - '+ str(i+1))
                #plt.spy(row[i].T)

        print('--- batch dataset visualized successfully ---')

    def visualize_all_dataset(self):
        print('--- Reading & visualizing Dataset -> dataset.npz file ---')
        compressed_dataset = np.load(self.dataset_path + 'dataset_all.npz')
        for row in range(0, 4887):
            data = compressed_dataset['arr_0'][row]
            plt.matshow(data.todense().T)
            plt.suptitle('frame - ' + str(row), fontsize=16)
            plt.show()
        print('--- dataset visualized successfully ---')
