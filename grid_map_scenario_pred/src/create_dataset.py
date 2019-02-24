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

class CreateDataset():
    def __init__(self):
        self.conf = Configuration()
        #self.dataset_path = os.path.join(self.conf.DATASET_SAVING_PATH, 'seq_10_samples/')
        #self.dataset_path = os.path.join(self.conf.DATASET_SAVING_PATH, 'seq_samples/')
        self.dataset_path = os.path.join(self.conf.DATASET_SAVING_PATH, 'samples_4frames_4ts/')
        if os.path.exists(self.dataset_path):
            self.conf.IS_DATASET_PATH_EXISTS == True
        else:
            self.handle_dir(self.dataset_path)

        for fname in os.listdir(self.dataset_path):
            if fname.endswith('.npz'):
                self.conf.IS_DATASET_EXISTS = True
                #print(self.conf.IS_DATASET_EXISTS)not
                break

        self.numpy_files = self.numpy_sorted_files()
        #self.seq_len = 10
        #self.sum_seq_lower = (self.seq_len - 1) * 75 - 20
        #self.sum_seq_upper = (self.seq_len - 1) * 75 + 20+1
        self.time_step_ahead = 4
        self.num_of_feat_label_frames = 4
        self.sum_feat_label_frames_lower = ((self.num_of_feat_label_frames - 1) * self.time_step_ahead) * 75 - 20
        self.sum_feat_label_frames_upper = ((self.num_of_feat_label_frames - 1) * self.time_step_ahead) * 75 + 20 + 1

        #print(str(len(self.numpy_files)))

    def handle_dir(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    # get desired numpy file
    def get_numpy_file(self, file_name):
        for filename in sorted(os.listdir(self.conf.IMAGE_ARRAY_PATH)):
            f = basename(filename).replace(".npy", "")
            #if float(f) == file_name:
            if int(f) == file_name:
               #print('filename> ' + str(file_name) + ' basename> '+ f)
               return filename

    def numpy_sorted_files(self):
        file_list = []
        for filename in sorted(os.listdir(self.conf.IMAGE_ARRAY_PATH)):
            fn = filename.replace(".npy", "")
            #file_list.append(float(fn))
            file_list.append(int(fn))
        sorted_list = sorted(file_list)
        #print(sorted_list)
        return sorted_list
        #print(sorted_list)

    # creating dataset by making combination of [feat1, feat2, feat3, label1]
    # from self.numpy_files. All the np arrays are converted to sparse matrix.
    # And then save it to the list array. After all the combination is done and
    # addded to the list (converted sparse matrix), then it is saved as npz format.
    def make_dataset(self):
        frames = []
        train = []
        test = []
        temp_train = []
        temp_test =[]
        data_row = []
        batch_row = []
        all_rows = []
        N = self.conf.NUMBER_OF_IMAGES_ARRAY # 4887
        counter = 0
        batch = 0
        framesetcount = 0
        batch_set_arr = []
        # SLIDING_FEATURE_BY_FRAME_WINDOW_FACTOR = 1 and SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR =1 produces 4776 combination
        for x in range(0, N-3):
            for i in range(x+1, N-2):
                if i - x == self.conf.SLIDING_FEATURE_BY_FRAME_WINDOW_FACTOR:
                    for j in range(i+1, N-1):
                        if j - i == self.conf.SLIDING_FEATURE_BY_FRAME_WINDOW_FACTOR:
                            for k in range(j+1, N):
                                if k - j <= self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR:
                                    # STARTING: Dataset preparing for visualization | code for saving each np array to one singe array.
                                    # Below code snippet will store each np array/gridmap to frames array.
                                    feat = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[x]))
                                    # converting np array to sparse matrix.not
                                    feat_sparse = sparse.csr_matrix(feat)
                                    frames.append(feat_sparse)
                                    if x == N-4:
                                        feat = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[N-3]))
                                        feat_sparse = sparse.csr_matrix(feat)
                                        frames.append(feat_sparse)
                                        feat = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[N-2]))
                                        feat_sparse = sparse.csr_matrix(feat)
                                        frames.append(feat_sparse)
                                        feat = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[N-1]))
                                        feat_sparse = sparse.csr_matrix(feat)
                                        frames.append(feat_sparse)
                                    # END: Dataset preparing for visualization | code for saving each np array to one singe array.

                                    # making a sample row by adding 4 np arrays (3 features + 1 label)
                                    # calculating difference time stamp between each np arrays.
                                    d1 = self.numpy_files[i] - self.numpy_files[x]
                                    d2 = self.numpy_files[j] - self.numpy_files[i]
                                    d3 = self.numpy_files[k] - self.numpy_files[j]
                                    sum = d1 + d2 + d3
                                    framesetcount += 1
                                    # checking set of elements boundary checking.
                                    # The first numpy to 4th numpy array range should be between 206ns to 245ns
                                    if sum > 205 and sum < 246:
                                        counter += 1
                                        feat1 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[x]))
                                        feat2 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[i]))
                                        feat3 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[j]))
                                        label1 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[k]))

                                        # converting np array to sparse matrix
                                        feat1_sparse = sparse.csr_matrix(feat1)
                                        feat2_sparse = sparse.csr_matrix(feat2)
                                        feat3_sparse = sparse.csr_matrix(feat3)
                                        label1_sparse = sparse.csr_matrix(label1)
                                        # add to list 4 np arrays as 3 features and 1 label combination.
                                        # data_row array always contains 1 sample row.
                                        data_row.append([feat1_sparse, feat2_sparse, feat3_sparse, label1_sparse])
                                        #batch_row.append([feat1_sparse, feat2_sparse, feat3_sparse, label1_sparse])
                                        # all_rows array contains all the sample rows
                                        all_rows.append([feat1_sparse, feat2_sparse, feat3_sparse, label1_sparse])
                                        temp = [x, i, j, k]
                                        print('counter - ' + str(counter) + ' >> ' + str(temp) + ' sum range ns of elem[1-4]:' + str(sum))
                                        print('data_row length: ', str(len(data_row)))
                                        self.save_sample(data_row, counter)
                                        data_row = []

                                        # to split dataset into train and test, train + test array used.
                                        if counter <= 3824:
                                            # train dataset 80% - 3824 rows
                                            train.append([feat1_sparse, feat2_sparse, feat3_sparse, label1_sparse])
                                        else:
                                            # test dataset 20% - 952 rows
                                            test.append([feat1_sparse, feat2_sparse, feat3_sparse, label1_sparse])

                                        # for testing in cpu - small train/test dataset created
                                        if counter <= 40:
                                            temp_train.append([feat1_sparse, feat2_sparse, feat3_sparse, label1_sparse])
                                        if counter > 40 and counter < 49:
                                            temp_test.append([feat1_sparse, feat2_sparse, feat3_sparse, label1_sparse])


                                        #temp = [numpy_files[x], numpy_files[i], numpy_files[j], numpy_files[k]]
                                        #print(temp)
                                        #print('set ' + str(framesetcount) + ' >> ' + str(temp) + ' sum range ns of elem[1-4]:' + str(sum))
                                        '''
                                        batch_set_arr.append(framesetcount)
                                        if len(batch_row) == 8:
                                            batch = batch + 1
                                            print('batch no - ' + str(batch) + ' | batch length> ' + str(len(batch_row)))
                                            print(batch_set_arr)
                                            print('Saving Dataset Batch - ' + str(batch))
                                            self.save_batch_dataset(batch_row, batch)

                                            batch_row = []
                                            batch_set_arr = []
                                            '''

        #print('array 0 th> ', data_row[0])
        #print('print array 40 element>', data_row[0][0].todense())
        print(str(len(train)))
        self.save_dataset(train, 'train')
        print(str(len(test)))
        self.save_dataset(test, 'test')
        print(str(len(temp_train)))
        self.save_dataset(temp_train, 'temp_train')
        print(str(len(temp_test)))
        self.save_dataset(temp_test, 'temp_test')

        print('Saving all 4776 sample rows in 1 npz file...')
        self.save_dataset(all_rows, self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR)
        print('Saving 4887 numpy files to 1 npz file...')
        self.save_dataset(frames, 'all')

        '''
        data_array = np.asarray(data_row)
        # saving whole sparse matrix list (converted to np array again) to compressed npz file
        np.savez_compressed(self.dataset_path + 'dataset_' + str(self.conf.SLIDING_LABEL_BY_FRAME_WINDOW_FACTOR), data_array)
        print('--- Dataset -> dataset.npz is saved ---')'''

    def save_batch_dataset(self, data_list, batchno):
        data_array = np.asarray(data_list)
        # for batch saving...
        np.savez_compressed(self.dataset_path + 'dsbatch_' + str(batchno), data_array)
        print('--- Batch Dataset is saved ---')

    def save_dataset(self, data_list, name):
        data_array = np.asarray(data_list)
        # saving whole sparse matrix list (converted to np array again) to compressed npz file
        np.savez_compressed(self.dataset_path + 'dataset_' + str(name), data_array)
        print('--- Dataset is saved as npz format ---')

    def save_sample(self, sample, sampleno):
        data_array = np.asarray(sample)
        # for each sample saving...
        np.savez_compressed(self.dataset_path + 'sample_' + str(sampleno).zfill(4), data_array)
        print('--- sample-' + str(sampleno).zfill(4) + ' is saved ---')

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

    # creating sequence dataset by making combination of [feat1, feat2, feat3, feat4, feat5, label1, label2, label3, label4, label5]
    # from self.numpy_files. All the np arrays are converted to sparse matrix.
    # And then save it to the list array. After all the combination is done and
    # addded to the list (converted sparse matrix), then it is saved as npz format.
    def make_seq_dataset(self):
        data_row = []
        N = self.conf.NUMBER_OF_IMAGES_ARRAY # 4887
        counter = 0
        seq_len = self.seq_len
        padding = 1
        counter = 0

        for x in range(0, N-(seq_len*padding)+1):
            np_files = []
            temp = []
            for i in range(x, x + (seq_len * padding), padding):
                np_files.append(self.numpy_files[i])
                temp.append(i)

            sum = 0
            for s in range(1, len(np_files)):
                sum += np_files[s] - np_files[s-1]

            if sum > self.sum_seq_lower and sum < self.sum_seq_upper:
                counter += 1
                elem_sparse = []
                for feat in range(0, len(np_files)):
                    elem = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(np_files[feat]))
                    elem_sparse.append(sparse.csr_matrix(elem))

                data_row.append(elem_sparse)
                print('counter - ' + str(len(np_files)) + ' >> ' + str(temp) + ' sum range ns of elem[1-6]:' + str(sum))
                print('data_row length: ', str(len(data_row)))
                self.save_sample(data_row, counter)
                data_row = []

    # creating dataset by making combination of [feat1, feat2, feat3, feat4, label1]
    # from self.numpy_files. All the np arrays are converted to sparse matrix.
    # And then save it to the list array. After all the combination is done and
    # addded to the list (converted sparse matrix), then it is saved as npz format.
    def make_time_step_ahead_dataset(self):
        data_row = []
        N = self.conf.NUMBER_OF_IMAGES_ARRAY # 4887
        counter = 0
        seq_len = self.num_of_feat_label_frames
        padding = self.time_step_ahead
        counter = 0
        loop_length = N - (seq_len*padding) + padding
        print(loop_length)

        for x in range(0, loop_length):
            np_files = []
            temp = []
            for i in range(x, x + (seq_len * padding), padding):
                np_files.append(self.numpy_files[i])
                temp.append(i)
            print(temp)

            sum = 0
            for s in range(1, len(np_files)):
                sum += np_files[s] - np_files[s-1]
                #print(sum)

            if sum > self.sum_feat_label_frames_lower and sum < self.sum_feat_label_frames_upper:
                counter += 1
                elem_sparse = []
                for feat in range(0, len(np_files)):
                    elem = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(np_files[feat]))
                    elem_sparse.append(sparse.csr_matrix(elem))

                data_row.append(elem_sparse)
                print('counter - ' + str(len(np_files)) + ' >> ' + str(temp) + ' sum range ns of elem[1-4]:' + str(sum))
                #print('data_row length: ', str(len(data_row)))
                self.save_sample(data_row, counter)
                data_row = []
            else:
                print('Not saved - ' + str(len(np_files)) + ' >> ' + str(temp) + ' sum range ns of elem[1-4]:' + str(sum))


    def create_dataset(self):
        #self.process_dataset()
        if self.conf.IS_MULTIPLE_DATASET_ALLOWED:
            #self.make_dataset()
            #self.make_seq_dataset()
            self.make_time_step_ahead_dataset()
            print('--- datasets created successfully ---')
        else:
            if self.conf.IS_DATASET_EXISTS:
                print('---datset already exists---')
            else:
                #self.make_dataset()
                #self.make_seq_dataset()
                self.make_time_step_ahead_dataset()
                print('--- dataset created successfully ---')


    '''
    for i in range(len(self.numpy_files)-3):
        feat1 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[i]))
        feat2 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[i+1]))
        feat3 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[i+2]))
        label1 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(self.numpy_files[i+3]))
        a = sparse.csr_matrix(feat1)
        b = sparse.csr_matrix(feat2)
        c = sparse.csr_matrix(feat3)
        d = sparse.csr_matrix(label1)
        train_data.append([a, b, c, d])
    print('array 0 th> ', train_data[0])
    #print('print array 40 element>', train_data[0][0].todense())
    print(str(len(train_data)))
    data_array = np.asarray(train_data)
    np.savez_compressed(self.dataset_path + 'dataset', data_array)
    print('--- Dataset -> dataset.npz is saved ---')'''

    '''# creating dataset batch and saving it as pickle formatself.
    # dataset batch format [feature1, feature2, feature3, label]
    def process_dataset(self):
        batch = 0
        no_of_iterations = 0
        train_data = []
        for i in [j + self.conf.NUMBER_OF_NUMPY_FILES_PER_HALF_SEC_INTERVAL for j in range(0, self.conf.RANGE_OF_FEATURE_1_NUMBER_FOR_ITERATION, self.conf.NUMBER_OF_NUMPY_FILES_PER_HALF_SEC_INTERVAL)]:
            # Rearranging numpy files as {feature1 = x, feature2 = y, feature3 =z}
            x = i
            y = i + self.conf.NUMBER_OF_NUMPY_FILES_PER_HALF_SEC_INTERVAL
            z = i + self.conf.NUMBER_OF_NUMPY_FILES_PER_HALF_SEC_INTERVAL * 2

            feat1 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(x))
            feat2 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(y))
            feat3 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(z))
            no_of_iterations += 1

            #print(len(train_data))
            # below loop finds 20 distinct label values (total 10 sec line each 0.5 sec interval) for above feature set {feat1, feat2, feat3}
            for elem in range(z + self.conf.NUMBER_OF_NUMPY_FILES_PER_HALF_SEC_INTERVAL, z + (self.conf.NUMBER_OF_NUMPY_FILES_PER_HALF_SEC_INTERVAL * (self.conf.NUMBER_OF_PREDICTION_INTERVAL +1)), 6):
                label1 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(elem))
                # add {feat1, feat2, feat3, label1} combination as array element to train_data list array
                train_data.append([feat1, feat2, feat3, label1])
                print('current array element->[' + str(x) + ', ' + str(y) + ', ' + str(z) + ', ' + str(elem)+']')

            # print('current data set (row number):' + str(len(train_data)))
            # After 5 iterations, getting a batch of 100 rows of train_data[] to save it as a pickle file.
            if no_of_iterations % 5 == 0:
                batch += 1
                datapart = np.asarray(train_data)
                path_and_filename_to_save = self.dataset_path + 'batch_'+ str(batch) + '_label_' + str(elem) + '.pickle'
                with open(path_and_filename_to_save, 'wb') as picklesave:
                    pickle.dump(datapart, picklesave)
                    print('saving pickle data file-> batch_data_' + str(batch) + '_label_' + str(elem) + '.pickle')

            # empty train_data array
            train_data[:] = []
            print(len(train_data))
        # break here for 40 batches (40 * 576 mb = 23GB data)
        if batch == 40:
            break

    # to test that pickle file can be read properly.
    def read_dataset(self):
        with open(self.dataset_path + 'batch_1_label_162.pickle', 'rb') as data:
            t_data = pickle.load(data)
            #t_data = data.read()
            #print(t_data)
            print(t_data[0])
            print('first feature element: ', t_data[0][0])
            print('shape = ', t_data.shape)

    def update_value(self, val):
        #global init_feat1_num, init_label_num, init_feat2_num, init_feat3_num
        self.conf.INIT_FEATURE_1_IMAGE_NUMBER += val
        self.conf.INIT_FEATURE_2_IMAGE_NUMBER += val
        self.conf.INIT_FEATURE_3_IMAGE_NUMBER += val
        self.conf.INIT_LABEL_IMAGE_NUMBER += val

    def process_dataset(self, label_num, feat1_num, feat2_num, feat3_num, label_range, filename_index):
        #for elem in range(NUMBER_OF_NUMPY_IMAGES - separate_time_slot - 81):
        for elem in range(label_num, label_num + label_range):
            #train_data = np.array([], dtype=np.uint8)
            train_data = []
            #train_data = np.array([])
            #np.empty((0,4), dtype=np.uint8)
            label1 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file( elem +
                             self.conf.TIME_DIFFERENCE))
            for i in range(self.conf.FEATURE_ITERATION_RANGE):
                feat1 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(i + feat1_num))
                for j in range(self.conf.FEATURE_ITERATION_RANGE):
                    feat2 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(j + feat2_num))
                    for k in range(self.conf.FEATURE_ITERATION_RANGE):
                        feat3 = np.load(self.conf.IMAGE_ARRAY_PATH + self.get_numpy_file(k + feat3_num))
                        train_data.append([feat1, feat2, feat3, label1])
                        #np.append(train_data, [[feat1, feat2, feat3, label1]], axis=0)
                        print('current array element->' + str(i + feat1_num) + ' ' + str(j+ feat2_num) + ' '
                               + str(k + feat3_num) + ' ' + str(elem + self.conf.TIME_DIFFERENCE))

            #print('current data set (row number):' + str(len(train_data)))
            datapart = np.asarray(train_data)
            path_and_filename_to_save = self.dataset_path + 'data_'+ str(filename_index) + '_' + str(elem) + '.pickle'
            with open(path_and_filename_to_save, 'wb') as picklesave:
                pickle.dump(datapart, picklesave)
                print('saving pickle data file-> data_' + str(filename_index) + '_' + str(elem) + '.pickle')


    def create_dataset(self):
        for i in range(0, self.conf.DATASET_BATCH_RANGE):
            val = 0
            self.update_value(i+val)
            self.process_dataset(self.conf.INIT_LABEL_IMAGE_NUMBER, self.conf.INIT_FEATURE_1_IMAGE_NUMBER,
                 self.conf.INIT_FEATURE_2_IMAGE_NUMBER, self.conf.INIT_FEATURE_3_IMAGE_NUMBER, self.conf.LABEL_ITERATION_RANGE, i)
            val += self.conf.UPDATE_FEATURE_LABEL_NUMBER
        print('--- dataset created. ---')'''
