import os
import shutil
from os import listdir
from os.path import isfile, join
import time
import cv2


def handle_out_folder(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def write_image_disk(path, img):
    cv2.imwrite(path, img)


def alphanum_key(s):
    x = s.replace("grid_map_", "").replace(".npy", "").replace(".jpg", "").replace(".png", "")
    # x = x.split("_")[0]
    return int(x)

def get_image_paths(folder_path):
    file_paths = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    file_paths.sort(key=alphanum_key)
    return file_paths

def timer(func):
    def inner_func(*args, **kwargs):
        current_time = time.clock()
        result = func(*args, **kwargs)
        print("The function {} took {} seconds".format(func.__name__, time.clock()-current_time))
        return result
    return inner_func

def add_trailing_slash(x):
    if x.endswith('/'):
        return x
    else:
        return x + '/'