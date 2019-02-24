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

np.set_printoptions(threshold=np.nan)

class CleanDataset():
    def __init__(self):
        self.conf = Configuration()
        self.dataset_path = os.path.join(self.conf.DATABASE_PATH, 'database/')
        self.numpy_files = self.numpy_sorted_files()
        self.IS_CLEAN_DATASET_DONE = False
        #print(str(len(self.numpy_files)))
        if not os.path.exists(self.dataset_path):
            self.handle_dir(self.dataset_path)
            print('cleaning data directory is created.')
        else:
            print('cleaning data directory exists.')
            self.IS_CLEAN_DATASET_DONE = True

    def handle_dir(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    # get desired numpy file
    def get_numpy_file(self, file_name):
        for filename in sorted(os.listdir(self.conf.NUMPY_RAW_PATH)):
            f = basename(filename).replace(".npy", "")
            #if float(f) == file_name:
            if int(f) == file_name:
               #print('filename> ' + str(file_name) + ' basename> '+ f)
               return filename

    def numpy_sorted_files(self):
        file_list = []
        for filename in sorted(os.listdir(self.conf.NUMPY_RAW_PATH)):
            fn = filename.replace(".npy", "")
            #file_list.append(float(fn))
            file_list.append(int(fn))
        sorted_list = sorted(file_list)
        return sorted_list
        #print(sorted_list)

    def pre_process_dataset(self):
        print('--- data cleaning started ---')
        diffs = []
        del_files = []
        for i in range(0, len(self.numpy_files)):
            if i+1 < len(self.numpy_files):
                diff = self.numpy_files[i+1] - self.numpy_files[i]
                if diff > 15:
                    diffs.append(diff)
                    shutil.copy(self.conf.NUMPY_RAW_PATH + str(self.numpy_files[i]) + '.npy', self.dataset_path + str(self.numpy_files[i]) + '.npy')
                else:
                    del_files.append(self.numpy_files[i+1])

        print(len(diffs))
        print(diffs)
        print(len(del_files))
        print(del_files)

    def clean_dataset(self):
        print(self.clean_dataset)
        if not self.IS_CLEAN_DATASET_DONE:
            self.pre_process_dataset()
        print('--- dataset cleaned successfully ---')
