#-*- coding: UTF-8 -*-
from multiprocessing import Process, Queue
import os
import time
import cv2
import numpy as np
import logging
import os
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image, ImageOps
from PIL import ImageFilter

import _init_paths

from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from utils.mio import load_string_list
from utils.image_utils import rescale_img, label_extract, label_to_array
from utils.partition import DataPartitioner

def get_XY(image):
    item = [0,0]
    if image is not None:
        item[0] = image.shape[0] ## height
        item[1] = image.shape[1] ## width
    return item

# 本次修改，主要集中在产生的日志文件没有两个了，只有一个log结尾的
# 如果用bash启动，那么就产生一个和result目下一样的日志文件
# 如果直接python启动，那么就没有日志文件，result下的目录之自动产生
# 就不能和日志一致了
# 推荐用bash train.sh启动
if __name__ == '__main__':
    cfg_file = '../configs/dist.yml'
    cfg_from_file(cfg_file)

    g_time = time.time()
    data_patition = DataPartitioner(2)
    data_param = data_patition.use()
    train_set = data_param[0]
    val_set = data_param[1]

    epoch_sum = len(train_set)
    starttime = time.time()
    count = 0
    for count,traindata in enumerate(train_set):
        imgs,labels=traindata
        img_num = 0
        for img in imgs:
            img_num += 1
            raw_image_hw = get_XY(img)
            #print('TRAIN::image_size = (%d,%d)' % (raw_image_hw[0], raw_image_hw[1]))
        #time.sleep(0.3)
        endtime = time.time()
        print("TRAIN::runtime:%.3f  epoch:%d/%d   This epoch images num:%d" % (endtime - starttime, count+1, epoch_sum, img_num ))
        starttime = endtime
    print("TRAIN::all runtime:%.3f"%(time.time()-g_time))  

    g_time = time.time()
    epoch_sum = len(val_set)
    starttime = time.time()
    for count,valdata in enumerate(val_set):
        imgs,labels=valdata

        img_num = 0
        for img in imgs:
            img_num += 1
            raw_image_hw = get_XY(img)
            #print('VAL::image_size = (%d,%d)' % (raw_image_hw[0], raw_image_hw[1]))
        #time.sleep(0.3)
        endtime = time.time()
        print("VAL::runtime:%.3f  epoch:%d/%d   This epoch images num:%d" % (endtime - starttime, count+1, epoch_sum, img_num ))
        starttime = endtime
    print("VAL::all runtime:%.3f"%(time.time()-g_time))  

