import numpy as np
#import cv2
import scipy.misc
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
from os import listdir, makedirs
from os.path import isfile, join, dirname, exists, basename
import shutil

from numpy import exp, array, random, dot

ROOT_PATH = '/home/mushfiqrahman/dataset/db_numpy/renaming/numpy_milisec/'

NUMPY_RAW_PATH = '/home/mushfiqrahman/dataset/db_numpy/numpy_raw_db/'
NUMPY_RAW_PATH_0 = os.path.join(NUMPY_RAW_PATH, 'numpy_mp10_0/')
NUMPY_RAW_PATH_1 = os.path.join(NUMPY_RAW_PATH, 'numpy_mp10_1/')
NUMPY_RAW_PATH_2 = os.path.join(NUMPY_RAW_PATH, 'numpy_mp10_2/')
NUMPY_RAW_PATH_3 = os.path.join(NUMPY_RAW_PATH, 'numpy_mp10_3/')
NUMPY_RAW_PATH_4 = os.path.join(NUMPY_RAW_PATH, 'numpy_mp10_4/')
NUMPY_RAW_PATH_5 = os.path.join(NUMPY_RAW_PATH, 'numpy_mp10_5/')
NUMPY_RAW_MERGED_PATH = os.path.join(NUMPY_RAW_PATH, 'merged_numpys_db/')
NUMPY_MILISEC_MERGED_PATH = os.path.join(NUMPY_RAW_PATH, 'merged_db/')
NUMPY_RAW_PATHS = [NUMPY_RAW_PATH_0, NUMPY_RAW_PATH_1, NUMPY_RAW_PATH_2, NUMPY_RAW_PATH_3, NUMPY_RAW_PATH_4, NUMPY_RAW_PATH_5]
#NUMPY_RAW_PATHS = [os.path.join(NUMPY_RAW_PATH, 'test_0/'), os.path.join(NUMPY_RAW_PATH, 'test_1/')]


#ROOT_PATH = '/home/mushfiqrahman/dataset/db_numpy/backup/numpy_sorted/'
# 4998-1512136280_516865690-1512136293_714657127
# first-> 0-1512135906_676934758-1512135903_225427038
# last-> 4999-1512136280_577886668-1512136293_789349910

def handle_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def sorted_files():
    sec_list = []
    for filename in sorted(os.listdir(ROOT_PATH)):
        filen = filename.replace(".npy", "")
        sec_list.append(float(filen))
    sorted_list = sorted(sec_list)
    print(sorted_list)

def converting_merged_files_to_milisec():
    start_sec = 1512135903
    if not os.path.exists(NUMPY_MILISEC_MERGED_PATH):
        handle_dir(NUMPY_MILISEC_MERGED_PATH)
        print('direcotry created for saving miliseconds file...')

    print('Saving converting milisecods merged files to '+ NUMPY_MILISEC_MERGED_PATH  + '...')
    # Conversion of radar_nano_seconds to radar_mili_seconds by taking integer division
    # of radar_nano_seconds/1000000 and, later, adding that to radar second. So, in this
    # way duplicate milisecond will be erased.
    count = 0
    for filename in sorted(os.listdir(NUMPY_RAW_MERGED_PATH)):
        #print(filename)
        filen = filename.replace(".npy", "")
        f = filen.split('-',2)
        #print('split f> ' + str(f))
        mili_f = f[2].split('_', 1)
        #print('split mili_f > ' + str(mili_f))

        # taking difference between statring seconds and current radar seconds.
        diff_sec = int(mili_f[0]) - start_sec
        #print('diff_sec>' + str(diff_sec))

        # converting radar_nano_seconds to miliseconds by taking integer division
        # of radar_nano_seconds/1000000. For duplicate miliseconds elimination,
        # integer division result is taken.
        #ms = float(mili_f[1])/1000000
        ms = int(mili_f[1])/1000000
        #print('mili second> ' + str(ms))

        # converting radar seconds to radar milisecond by multipling 1000 and then
        # adding it with converted radar_mili_seconds from radar_nano_seconds
        #total_ms = ms + float(diff_sec * 1000)
        total_ms = ms + int(diff_sec * 1000)
        #print('total>' + str(total_ms))

        new_name = filename.replace(filen, str(total_ms))
        if not os.path.exists(NUMPY_MILISEC_MERGED_PATH + new_name):
            shutil.copy(NUMPY_RAW_MERGED_PATH + filename, NUMPY_MILISEC_MERGED_PATH + new_name)
            count = count + 1
            print(str(count) + " -> "+ filename + ' is copied to -> ' + new_name)
        else:
            print(filename + ' is already exists.')

        #os.rename(os.path.join( ROOT_PATH, filename), os.path.join( ROOT_PATH, new_name))
    print(str(count) + ' files are copied. Conversion to miliseconds are done.')

# merging all the numpys from mp10 processed directory which is generated as output
# from ros2roh node. Currently, 6 direcotries are merging withiin 1 merged directory sortedly.
def merging_all_numpys_sortedly_by_copying(index, path):
    count = index
    print('copying file starting from index ' + str(index+1) + '...')
    for filename in sorted(os.listdir(path)):
        #print('filename> ', filename)
        f = filename.split('-',1)[0]
        #print(f)
        count =  count + 1
        val = int(f) + index + 1
        #print(str(val))
        new_name = filename.replace(f, str(val), 1)
        #print('new_name> ', new_name)
        shutil.copy(path + filename, NUMPY_RAW_MERGED_PATH + new_name)
        #print(filename + ' is copied to -> ' + new_name)
        #os.rename(os.path.join( ROOT_PATH, filename), os.path.join( ROOT_PATH, new_name))
    #print('count>', count)
    return count

# processing merging operation over all mp10 directory files
def merging_all_numpys():
    cur_index = 0
    for i in range(0, len(NUMPY_RAW_PATHS)):
        print('starting mp10_' + str(i) + ' numpys to be merged into ' + NUMPY_RAW_MERGED_PATH + '...')
        previous_index = cur_index
        cur_index = merging_all_numpys_sortedly_by_copying(cur_index, NUMPY_RAW_PATHS[i])
        print('Total ' + str(cur_index - previous_index) + ' files are copied from mp10_'+ str(i) + '. Finished at index ' + str(cur_index) + '.\n')

    print('Merging done. All mp10 files are merged to ' + NUMPY_RAW_MERGED_PATH)


#current_index = 880
#merging_all_numpys()
converting_merged_files_to_milisec()
#sorted_files()
