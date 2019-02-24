import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os


class Configuration():
    def __init__(self):
        self.NUMPY_RAW_PATH = '/home/mushfiqrahman/dataset/db_numpy/numpy_raw_db/merged_db/'
        self.IMAGE_ARRAY_PATH = '/home/mushfiqrahman/dataset/db_numpy/numpy_raw_db/database/'
        self.DATABASE_PATH = '/home/mushfiqrahman/dataset/db_numpy/numpy_raw_db/'
        #self.IMAGE_ARRAY_PATH = '/home/mushfiqrahman/dataset/db_numpy/test/'
        self.DATASET_SAVING_PATH = '/home/mushfiqrahman/dataset/db_numpy/'
        # total number of numpy files collected from mp10 bag files by using ros2roh node
        self.NUMBER_OF_IMAGES_ARRAY = 4887
        # Number of iteration for {feat1, feat2, feat3, label1} combination of 5000 arrays.
        self.NUMBER_OF_ITERATION = 4884
        # equal difference by ms between features {from one np frame to another frame}
        self.SLIDING_FEATURE_BY_FRAME_WINDOW_FACTOR = 1
        # difference by ms between feature3 and label1 {from last feature -> label frame}
        self.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR = 1
        self.IS_DATASET_EXISTS = False
        self.IS_DATASET_PATH_EXISTS = False
        self.IS_MULTIPLE_DATASET_ALLOWED = False
        self.IS_BATCH_DATASET_EXISTS = False
        self.SORTED_NP_FILES = self.numpy_sorted_files()

    def numpy_sorted_files(self):
        file_list = []
        for filename in sorted(os.listdir(self.IMAGE_ARRAY_PATH)):
            fn = filename.replace(".npy", "")
            file_list.append(int(fn))
        sorted_list = sorted(file_list)
        return sorted_list



        '''
        # 0.5 sec equivalent to numpy array file -> 6-1512135906_676934758-1512135903_676630589.npy, so it gives
        # 0.5 sec = 6. Hence, 1 sec approximately numpy array file no. 12
        self.NUMBER_OF_NUMPY_FILES_PER_HALF_SEC_INTERVAL = 6
        # For each {feature1, feature2, feature3} set, 20 distinct values are measured
        self.NUMBER_OF_PREDICTION_INTERVAL = 20
        self.INIT_FEATURE_1_IMAGE_NUMBER = 6
        self.INIT_FEATURE_2_IMAGE_NUMBER = 12
        self.INIT_FEATURE_3_IMAGE_NUMBER = 18
        self.INIT_LABEL_IMAGE_NUMBER = 24
        self.LABEL_ITERATION_RANGE = 5
        self.FEATURE_ITERATION_RANGE = 5 '''
