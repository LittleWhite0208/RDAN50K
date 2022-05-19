#-*- coding=utf-8 -*-
import logging
import os
import cv2
import numpy as np
import time
import threading
import shutil
import getpass
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import multiprocessing,Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate

import torch.utils.data as data
from PIL import Image, ImageOps
from PIL import ImageFilter
from core.config import cfg
from utils.mio import load_string_list
from utils.image_utils import rescale_img, label_extract, label_to_array

logger = logging.getLogger(__name__)

def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据


class BetaroadDataset30(data.Dataset):
    def __init__(self, name, _index_tuple):
        logger.debug('Creating: {}'.format(name))
        print(_index_tuple)
        self.name = name
        mgr = multiprocessing.Manager()
        self._index_dict = mgr.dict()
        self._data_tuple = _index_tuple
        line_sum = self.getLines(self._data_tuple[0]) 
        if self._data_tuple[2] > line_sum-1:
            self._data_tuple[2] = line_sum-1
        assert (self._data_tuple[1] < self._data_tuple[2]) and (self._data_tuple[1] >= 0)
        self._index_worker()


    #例子
    #https://www.jianshu.com/p/6e22d21c84be
    def __getitem__(self, index):
        #print('__getitem__(%d)'%index)
        # 注意这里加2的是一套，和初始化函数一致
        path_tuple = self._load_index(index)
        #print(path_tuple)
        if path_tuple is None:
            print('1 index = %d'%(index))
            return None,None

        # 读取标记文件
        label_grid = self.func_label_loader(path_tuple[2])
        #print(label_grid)
        if label_grid is None:
            print('2 label path=%s, index=%d'%(path_tuple[2],index))
            return None,None
        
         # 读取图片文件
        img_grid = self.func_image_loader(path_tuple[1])
        if img_grid is None:
            print('3 image path=%s, index=%d'%(path_tuple[1],index))
            return None,None
      
        # 文件名，处理后的图片，标签格子
        # return (path_tuple[0], img_grid, label_grid)
        return img_grid, label_grid

    def __len__(self):
        _len = self._data_tuple[2] - self._data_tuple[1] + 1
        #print('__len__=%d'%_len)
        return _len


    # 这个函数和_index_worker一样，但是是全部读取到内存中，看看速度是快一些
    # _data_tuple=(datafile, _index_start, _index_end)
    # 配套_load_index(self, idx)函数
    # 注意，这里需要含前后两个位置
    def _index_worker2(self):
        # 是仁义设计的txt文件，不是fileindex
        # 仁义确定的这个文件，第一列是jpg文件名，第二列是图片全路径，第三列是标记全路径
        idx_path = self._data_tuple[0]
        #print(self._data_tuple)
        self._index_dict.clear()

        count = 0
        effective_count = 0
        try:
            if os.path.exists(idx_path) is False:
                return

            f = open(idx_path, "r")
            #print('................................')
            while True:
                line = f.readline()
                count += 1
                if line == '':
                    return

                if count < self._data_tuple[1]:
                    continue

                if count > self._data_tuple[2]:
                    break
            
                self._index_dict[effective_count] = line
                effective_count += 1
            f.close()


        except IOError as err:
            pass
        print('effective_count is %d, count=%d.'%(effective_count,count))

    '''
    ## 一个函数，这里是第一套方案，需要的是线程
    def _index_worker(self):
        #开始读取索引，放到_index_dict中
        t=threading.Thread(target=self._index_run,)
        t.setDaemon(True)
        t.start()

    ## 这里filename_lists里面是tuple，三个元素，一个是txt，一个是
    ## 图的目录，一个是label的目录
    def _index_run(self):
        self._index_dict.clear()
        index = 0
        for pos in self._index_parse():
            self._index_dict[index] = pos
            index += 1
        print('in __index_run, index=%d,len=%d.'%(index, len(self._index_dict)))

    # 这个函数得到的是图片文件名和全路径的文件名对照
    # 注意，这里需要含前后两个位置
    # _data_tuple=(datafile, _index_start, _index_end)
    def _index_parse(self):
        # 是仁义设计的txt文件，不是fileindex
        # 仁义确定的这个文件，第一列是jpg文件名，第二列是图片全路径，第三列是标记全路径
        idx_path = self._data_tuple[0]
        count = 0
        effective_count = 0
        try:
            if os.path.exists(idx_path) is False:
                return

            f = open(idx_path, "r")
            pos = 0
            while True:
                line = f.readline()
                count += 1
                if line == '':
                    return
                cur_pos = f.tell()
                if count < self._data_tuple[1]:
                    continue

                if count > self._data_tuple[2]+1:
                    break
            
                yield pos
                pos = cur_pos
                effective_count += 1
            f.close()
        except IOError as err:
            pass
    '''

    # 这个函数得到的是图片文件名和全路径的文件名对照
    # 注意，这里需要含前后两个位置
    # _data_tuple=(datafile, _index_start, _index_end)
    def _index_worker(self):
        # 是仁义设计的txt文件，不是fileindex
        # 仁义确定的这个文件，第一列是jpg文件名，第二列是图片全路径，第三列是标记全路径
        idx_path = self._data_tuple[0]
        count = -1
        effective_count = 0
        try:
            if os.path.exists(idx_path) is False:
                return

            f = open(idx_path, "r")
            pos = 0
            while True:
                line = f.readline()
                count += 1
                next_pos = f.tell()
                if count < self._data_tuple[1]:
                    continue
                
                if count > self._data_tuple[2]:
                    raise EOFError
            
                self._index_dict[effective_count] = pos
                effective_count += 1
                pos = next_pos
            f.close()
        except IOError as err:
            pass
        except EOFError:
            f.close()
        print('in _index_worker, count=%d,len=%d.'%(count, len(self._index_dict)))

    # 这个函数得到的是图片文件名和全路径的文件名对照
    # 配合第二套方案用，全部放到内存，这样会快很多？
    # _data_tuple=(datafile, _index_start, _index_end)
    def _load_index2(self, idx):
        # 是仁义设计的txt文件，不是fileindex
        # 仁义确定的这个文件，第一列是jpg文件名，第二列是图片全路径，第三列是标记全路径
        try:
            line = self._index_dict[idx]
            fileTmp = line.replace('\r','').replace('\n','')
            file_meta = fileTmp.split(",")
            bound = len(file_meta)

            if bound < 3:
                return None

            img_file = os.path.join(file_meta[1], file_meta[0])
            label_file = os.path.join(file_meta[2], file_meta[0])
            label_file = label_file.replace('.jpg','.txt')
            # 这个需要注意：分量是，文件名，全路径图片名，全路径标记名
            return (file_meta[0], img_file, label_file)

        except IOError as err:
            return None

    # 这个函数得到的是图片文件名和全路径的文件名对照
    # _data_tuple=(datafile, _index_start, _index_end)
    def _load_index(self, idx):
        # 是仁义设计的txt文件，不是fileindex
        # 仁义确定的这个文件，第一列是jpg文件名，第二列是图片全路径，第三列是标记全路径
        idx_path = self._data_tuple[0]
        #print(idx_path)
        #print('_index_dict len = %d'%len(self._index_dict))
        try:
            if os.path.exists(idx_path) == False:
                return None

            wait_num = 5
            sleep_time = 0.1
            while idx not in self._index_dict:
                wait_num -= 1
                sleep_time += 0.1
                if wait_num > 0:
                    time.sleep(sleep_time)
                    continue
                else:
                    return None
                
            f = open(idx_path, "r")
            f.seek(self._index_dict[idx])
            line = f.readline()

            fileTmp = line.replace('\r','').replace('\n','')
            file_meta = fileTmp.split(",")
            bound = len(file_meta)

            if bound < 3:
                return None

            img_file = os.path.join(file_meta[1], file_meta[0])
            label_file = os.path.join(file_meta[2], file_meta[0])
            label_file = label_file.replace('.jpg','.txt')
            # 这个需要注意：分量是，文件名，全路径图片名，全路径标记名
            return (file_meta[0], img_file, label_file)

        except IOError as err:
            return None


    # 这里直接就是npy的数据，得到就是二维的数组
    def func_label_loader(self, label_filepath):
        try:
            if not os.path.exists(label_filepath):
                return None

            if '.txt' in label_filepath:
                return self._load_label(label_filepath)
            else:
                return np.load(label_filepath)
        except:
            pass
        return None

    def _load_label(self, label_filepath):
        # 载入人工标记的文本，k是文件名，v是全路径的jpg文件名
        # 读入一个人工标记的文件？标记可能很多，如果需要扩展成多种标签，那么扩展ret就可以
        try:
            label_list = []
            if os.path.exists(label_filepath):
                with open(label_filepath, "r", encoding='utf-8') as l:
                    lines = l.readlines()
                # 一个文件的标记就是一个数组
                #print(lines)
                label_list = self.extract_label(lines)
                #print(label_list)
            return self.label_grid(label_list)
        except Exception as err:
            return None

    def extract_label(self, lines):
        return self._extract_label_crack(lines)

    # 从标记文件中获取裂缝的坐标信息，解析成一个列表，这个
    # 列表的每个元素是一个tuple，每个tuple就是x,y坐标
    def _extract_label_crack(self, lines):
        ret = []
        for line in lines:
            if line.find(cfg.LABEL_TYPE.CRACK) != -1:
                ax = line.strip("\n").replace(cfg.LABEL_TYPE.CRACK+':', "").split(",")
                ax = tuple(int(x) for x in ax)
                ret.append(ax)
        #print(ret)
        return ret

    # 从标记文件中获取修补的坐标信息，解析成一个列表，这个
    # 列表的每个元素是一个tuple，每个tuple就是x,y坐标
    def _extract_label_repair(self, lines):
        ret = []
        for line in lines:
            if line.find(cfg.LABEL_TYPE.REPAIR) != -1:
                ax = line.strip("\n").replace(cfg.LABEL_TYPE.REPAIR+':', "").split(",")
                ax = tuple(int(x) for x in ax)
                ret.append(ax)
        return ret


    def label_grid(self,label):
        #print(label)
        #print('H:W=%d:%d'%(cfg.TRAIN.IMG_BLOCK_H, cfg.TRAIN.IMG_BLOCK_W))
        grid = np.zeros((cfg.TRAIN.IMG_BLOCK_H, cfg.TRAIN.IMG_BLOCK_W), np.uint8)
        # 将含有坐标的列表，转换成二维数组个，这个三位的，但是实际上是两个维度的
        if label is not None:
            for l in label:
                if l[0] <= cfg.TRAIN.IMG_BLOCK_H and l[1] <= cfg.TRAIN.IMG_BLOCK_W:
                    grid[l[0] - 1, l[1] - 1] = 1
        return grid


    def func_image_loader2(self, img_filepath):
        try:
            if not os.path.exists(img_filepath):
                return None
            img = Image.open(img_filepath).convert('L')
            img = img.crop((0,0,cfg.TRAIN.RAW_IMAGE_COLS,cfg.TRAIN.RAW_IMAGE_ROWS))
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((cfg.TRAIN.SHRINK_IMG_H, cfg.TRAIN.SHRINK_IMG_W)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((.5,.5,.5),(.5,.5,.5))
                ])
            img = transform(img)
            return img
        except:
            return None

    def func_image_loader(self, img_filepath):
        try:
            #print('1==='+img_filepath)
            if not os.path.exists(img_filepath):
                return None
            #print('2==='+img_filepath)
            img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
            #print('1-func_image_loader')
            return self.img_grid(img)
        except:
            return None

    def img_grid(self,img):
        if(img is None):
            img = np.zeros((cfg.TRAIN.RAW_IMAGE_ROWS,cfg.TRAIN.RAW_IMAGE_COLS))
            img[:]=255

        if cfg.TRAIN.RAW_IMAGE_ROWS < img.shape[0]:
            img = img[:cfg.TRAIN.RAW_IMAGE_ROWS, :]

        if cfg.TRAIN.RAW_IMAGE_COLS < img.shape[1]:
            img = img[:,:cfg.TRAIN.RAW_IMAGE_COLS]

        if (img.shape[0] > cfg.TRAIN.RAW_IMAGE_ROWS or img.shape[1] > cfg.TRAIN.RAW_IMAGE_COLS):
            img = pad(img, ((0, cfg.TRAIN.RAW_IMAGE_ROWS - img.shape[0]),(0, cfg.TRAIN.RAW_IMAGE_COLS - img.shape[1])), 'constant',constant_values=255)
        
        img = cv2.resize(img, (cfg.TRAIN.SHRINK_IMG_W, cfg.TRAIN.SHRINK_IMG_H), interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')  # 转换类型为float32
        img -= np.mean(img)
        img /= 255.
        return img

    def getLines(self, datafile):
        count = 0
        try:
            f = open(datafile, "r")
            for line in f:
                count += 1
            f.close()
        except IOError as err:
            print('getLines:File error:'+str(err))
        #print('file line number:%d'%count)
        return count

    ##  把扩展的维度值为1的，压缩回去
    def squeeze(self, label_mask):
        return np.squeeze(label_mask, axis=1)

    # 扩充维度，使其满足pytorch需要的shape，和
    def expand_dims(self, img):
        return np.expand_dims(img, axis=1)


