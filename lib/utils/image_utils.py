import cv2
import numpy as np
import math
from core.config import cfg

block_size = 50
pad = 25
downsample = 32.
patch_must_be_divided_by = 32
patch_size = 256





def label_extract(manual, label_type):
    with open(manual, 'r') as f:
        lines = f.readlines()
    label_list = []
    for line in lines:
        if line.find(label_type) != -1:
            ax = line.strip("\n").replace(label_type+':', "").split(",")
            ax = tuple( int(''.join(list(filter(str.isdigit, x)))) for x in ax)
            y,x = ax
            label_list.append([y-1, x-1])
    return label_list

def label_to_array(shape, label_list):
    label_array = np.zeros(shape)
    for item in label_list:
        y, x = item
        if y <= shape[0] and x <= shape[1]:
            label_array[y, x] = 1
    return label_array

def rescale_img(test_image, source_block_size = 50, target_block_size = 32):
    target_patch_size = 256
    # target_block_num = target_patch_size / target_block_size
    rescale_rate = target_block_size * 1. / source_block_size

    pad = source_block_size//2
    height, width = test_image.shape
    new_height = height + pad
    new_width = width + pad
    new_height -= (new_height % source_block_size)
    new_width -= (new_width % source_block_size)
    pad_img = np.zeros((new_height, new_width), dtype='uint8')
    pad_img[:height, :width] = test_image[:min(height, new_height), :min(width, new_width)]
    target_img = cv2.resize(pad_img, (int(new_width * rescale_rate), int(new_height * rescale_rate)))
    return test_image, np.array(target_img)

def get_patch_index(label_array, mask, patch_shape=8, stride=4, thresh=3):
    y_max, x_max = np.max(label_array, 0)
    y_min, x_min = np.min(label_array, 0)
    patch_index = []
    i = y_min
    while i + patch_shape <= y_max + stride:
        j = x_min
        while j + patch_shape <= x_max + stride:
            patch = mask[i:(i + patch_shape), j:(j + patch_shape)]
            if patch.sum() > thresh:
                patch_index.append((i, j))
            j += stride
        i += stride
    return patch_index