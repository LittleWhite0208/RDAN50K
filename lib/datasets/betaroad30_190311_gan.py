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
from numpy.lib import pad
from PIL import Image

logger = logging.getLogger(__name__)

def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    try:
        batch = default_collate(batch) # 用默认方式拼接过滤后的batch数据
    except Exception as err:
        pass
    return batch

class BetaroadDataset30(data.Dataset):
    def __init__(self, name, txtfile):
        logger.debug('Creating: {}'.format(name))
        self.name = name
        mgr = multiprocessing.Manager()
        self._index_dict = mgr.dict()
        line_sum = self.getLines(txtfile) 
        print('true line sum = %d.'%line_sum)
        self._data_tuple = (txtfile, 0, line_sum-1)
        print(self._data_tuple)
        self._index_worker2()


    #例子
    #https://www.jianshu.com/p/6e22d21c84be
    def __getitem__(self, index):
        #print('__getitem__(%d)'%index)
        # 注意这里加2的是一套，和初始化函数一致
        path_tuple = self._load_index2(index)
        #print(path_tuple)
        if path_tuple is None:
            print('1 index = %d'%(index))
            return None,None

        # 读取标记文件
        manual_path = path_tuple[2]
        if os.path.exists(manual_path):
            manual_label_file = open(manual_path, "r")
            label_lines = manual_label_file.readlines()
            manual_label_file.close()
            flag = False  # 判断人工是否修改过,默认没修改过
            for each_label_line in label_lines:
                if "Bad_SonSanBlockCount" in each_label_line or "Bad_KengCaoBlockCount" in each_label_line:
                    flag = True
            if flag == False:  # 人工没修改过，按机器识别的标记
                manual_label_crack = self.func_label_loader(manual_path, "auto", "crack")
                manual_label_repair = self.func_label_loader(manual_path, "auto", "repair")
            else:  # 人工修改过，按人工的标记
                manual_label_crack = self.func_label_loader(manual_path, "manual", "crack")
                manual_label_repair = self.func_label_loader(manual_path, "manual", "repair")
        else:
            manual_label_crack = None
            manual_label_repair = None

        label_grid = manual_label_crack
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
        img_grid=np.expand_dims(img_grid, axis=0)
        label_grid=np.expand_dims(label_grid, axis=0)
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
                if line == '':
                    break
                self._index_dict[count]=line
                count += 1
            f.close()
        except Exception as err:
            pass
        print('count=%d. len=%d.'%(count, len(self._index_dict)))

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
            fileTmp = line.replace('\r', '').replace('\n', '')
            file_meta = fileTmp.split(",")
            bound = len(file_meta)

            if bound < 3:
                print('bound < 3')
                return None

            img_file = os.path.join(file_meta[1], file_meta[0])
            label_file = os.path.join(file_meta[2], file_meta[0])
            if ".jpg" in label_file:
                label_file = label_file.replace('.jpg', '.txt')
            elif ".png" in label_file:
                label_file = label_file.replace('.png', '.txt')
            # 这个需要注意：分量是，文件名，全路径图片名，全路径标记名
            return (file_meta[0], img_file, label_file)
        except Exception as err:
            print('_load_index2 except Exception as err:%s' % str(err))
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
                print('1111')
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
                    print('22222')
                    return None
                
            f = open(idx_path, "r")
            f.seek(self._index_dict[idx])
            line = f.readline()

            fileTmp = line.replace('\r','').replace('\n','')
            file_meta = fileTmp.split(",")
            bound = len(file_meta)

            if bound < 3:
                print('333333')
                return None

            img_file = os.path.join(file_meta[1], file_meta[0])
            label_file = os.path.join(file_meta[2], file_meta[0])
            label_file = label_file.replace('.jpg','.txt')
            # 这个需要注意：分量是，文件名，全路径图片名，全路径标记名
            return (file_meta[0], img_file, label_file)

        except IOError as err:
            print('4444')
            return None

    # 这里直接就是npy的数据，得到就是二维的数组
    def func_label_loader(self,label_filepath, label_type, disease_type):
        try:
            if not os.path.exists(label_filepath):
                return None

            if '.txt' in label_filepath:
                return self._load_label(label_filepath, label_type, disease_type)
            else:
                return np.load(label_filepath)
        except:
            pass
        return None

    def _load_label(self,label_filepath, label_type, disease_type):
        # 载入人工标记的文本，k是文件名，v是全路径的jpg文件名
        # 读入一个人工标记的文件？标记可能很多，如果需要扩展成多种标签，那么扩展ret就可以
        try:
            label_list = []
            if os.path.exists(label_filepath):
                with open(label_filepath, "r", encoding='utf-8') as l:
                    lines = l.readlines()
                # 一个文件的标记就是一个数组
                # print(lines)
                label_list = self.extract_label(lines, label_type, disease_type)
                # print(label_list)
            return self.label_grid(label_list)
        except Exception as err:
            return None

    def extract_label(self,lines, label_type, disease_type):
        if disease_type == "crack":
            return self._extract_label_crack(lines, label_type)
        elif disease_type == "repair":
            return self._extract_label_repair(lines, label_type)

    # 从标记文件中获取裂缝的坐标信息，解析成一个列表，这个
    # 列表的每个元素是一个tuple，每个tuple就是x,y坐标
    def _extract_label_crack(self,lines, label_type):
        ret = []
        # print(label_type)
        # print(lines)
        if label_type == "manual":
            for line in lines:
                if line.find('Bad_BlockPos') != -1:
                    ax = line.strip("\n").replace('Bad_BlockPos' + ':', "").split(",")
                    ax = tuple(int(x) for x in ax)
                    ret.append(ax)
        elif label_type == "auto":
            for line in lines:
                if line.find('BadBlockXY') != -1:
                    ax = line.strip("\n").replace('BadBlockXY' + ':', "").split(",")
                    ax = tuple(int(x) for x in ax)
                    ret.append(ax)
        # print(ret)
        return ret

    # 从标记文件中获取修补的坐标信息，解析成一个列表，这个
    # 列表的每个元素是一个tuple，每个tuple就是x,y坐标
    def _extract_label_repair(self,lines, label_type):
        ret = []
        if label_type == "manual":
            for line in lines:
                if line.find('Bad_RepairBlockPos') != -1:
                    ax = line.strip("\n").replace('Bad_RepairBlockPos' + ':', "").split(",")
                    ax = tuple(int(x) for x in ax)
                    ret.append(ax)
        elif label_type == "auto":
            for line in lines:
                if line.find('BadRepairBlockXY') != -1:
                    ax = line.strip("\n").replace('BadRepairBlockXY' + ':', "").split(",")
                    ax = tuple(int(x) for x in ax)
                    ret.append(ax)
            # print(ret)
        return ret

    def label_grid(self,label):
        # print(label)
        # print('H:W=%d:%d'%(cfg.TRAIN.IMG_BLOCK_H, cfg.TRAIN.IMG_BLOCK_W))
        grid = np.zeros((22, 34), np.uint8)
        # 将含有坐标的列表，转换成二维数组个，这个三位的，但是实际上是两个维度的
        if label is not None:
            for l in label:
                if l[0] <= 22 and l[1] <= 34:
                    grid[l[0] - 1, l[1] - 1] = 1

        return grid

    def accuracy(self,output, target):
        y_pred_t = torch.gt(output, threshold).float()
        y_true_t = torch.gt(target, threshold).float()
        intersection = torch.sum(y_pred_t * y_true_t)  ##注意* 和 &的差异
        dice = (2.0 * intersection + smooth) / (torch.sum(y_true_t) + torch.sum(y_pred_t) + smooth)
        return dice.item()


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
            #img =self.read_RGBimage(img_filepath)
            #print('1-func_image_loader')
            #RGB模式或者单通道模式
            return self.img_grid2(img)
            #return self.RGBimg_grid(img)
        except:
            return None

    def RGBimg_grid(self, img):
        # 对img进行pading，resize操作  得到（3,647,1000)的图片
        if (img is None):
            img = np.zeros((3, cfg.TRAIN.RAW_IMAGE_ROWS, cfg.TRAIN.RAW_IMAGE_COLS))

        weight = np.shape(img[0])[1]  # 图片的长
        height = np.shape(img[0])[0]  # 图片的宽

        # print("weight:", weight)
        # print("height:", height)
        # 裁剪
        if (weight > cfg.TRAIN.RAW_IMAGE_COLS):
            weight = cfg.TRAIN.RAW_IMAGE_COLS
            img = img[:, :cfg.TRAIN.RAW_IMAGE_ROWS, :]
        if (height > cfg.TRAIN.RAW_IMAGE_ROWS):
            height = cfg.TRAIN.RAW_IMAGE_ROWS
            img = img[:, :, :cfg.TRAIN.RAW_IMAGE_COLS]

        # 填充
        img = img.transpose((1, 2, 0))
        # print(np.shape(img))
        channel_one = img[:, :, 0]
        channel_two = img[:, :, 1]
        channel_three = img[:, :, 2]

        channel_one = np.pad(channel_one, (
            (0, cfg.TRAIN.RAW_IMAGE_ROWS - height), (0, cfg.TRAIN.RAW_IMAGE_COLS - weight)), 'constant',
                             constant_values=(0, 0))
        channel_one -= np.mean(channel_one)
        channel_one /= 255.
        channel_two = np.pad(channel_two, (
            (0, cfg.TRAIN.RAW_IMAGE_ROWS - height), (0, cfg.TRAIN.RAW_IMAGE_COLS - weight)), 'constant',
                             constant_values=(0, 0))
        channel_two -= np.mean(channel_two)
        channel_two /= 255.

        channel_three = np.pad(channel_three, (
            (0,cfg.TRAIN.RAW_IMAGE_ROWS - height), (0, cfg.TRAIN.RAW_IMAGE_COLS - weight)), 'constant',
                               constant_values=(0, 0))
        channel_three -= np.mean(channel_three)
        channel_three /= 255.
        img = np.dstack((channel_one, channel_two, channel_three))
        # print("######")
        # print(np.shape(img))

        img = cv2.resize(src=img, dsize=(cfg.TRAIN.SHRINK_IMG_W,cfg.TRAIN.SHRINK_IMG_H))
        img = img.transpose((2, 0, 1))
        # print(np.shape(img))
        return img

    def read_RGBimage(self, path, dtype=np.float32, color=True):

        f = Image.open(path)
        try:
            if color:
                img = f.convert('RGB')
            else:
                img = f.convert('P')
            img = np.asarray(img, dtype=dtype)
        finally:
            if hasattr(f, 'close'):
                f.close()

        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            return img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1))


    def img_grid2(self,img):
        # print(img.shape)
        if(img is None):
            img = np.zeros((cfg.TRAIN.SHRINK_IMG_H,cfg.TRAIN.SHRINK_IMG_W))
            img[:]=255
        #
        # if cfg.TRAIN.RAW_IMAGE_ROWS < img.shape[0]:
        #     img = img[:cfg.TRAIN.RAW_IMAGE_ROWS, :]
        #
        # if cfg.TRAIN.RAW_IMAGE_COLS < img.shape[1]:
        #     img = img[:,:cfg.TRAIN.RAW_IMAGE_COLS]
        #
        #
        # img = pad(img, ((0, cfg.TRAIN.RAW_IMAGE_ROWS - img.shape[0]),(0, cfg.TRAIN.RAW_IMAGE_COLS - img.shape[1])), 'constant',constant_values=255)
        #
        # img = cv2.resize(img, (cfg.TRAIN.SHRINK_IMG_W, cfg.TRAIN.SHRINK_IMG_H), interpolation=cv2.INTER_CUBIC)

        img = img.astype('float32')  # 转换类型为float32
        img -= np.mean(img)
        img /= 255.
        # print(img.shape)
        return img

    def img_grid(self,img):
        # print(img.shape)
        if(img is None):
            img = np.zeros((cfg.TRAIN.RAW_IMAGE_ROWS, cfg.TRAIN.RAW_IMAGE_COLS))
            img[:]=255

        if cfg.TRAIN.RAW_IMAGE_ROWS < img.shape[0]:
            img = img[:cfg.TRAIN.RAW_IMAGE_ROWS, :]

        if cfg.TRAIN.RAW_IMAGE_COLS < img.shape[1]:
            img = img[:,:cfg.TRAIN.RAW_IMAGE_COLS]


        img = pad(img, ((0, cfg.TRAIN.RAW_IMAGE_ROWS - img.shape[0]),(0, cfg.TRAIN.RAW_IMAGE_COLS - img.shape[1])), 'constant',constant_values=255)

        img = cv2.resize(img, (cfg.TRAIN.SHRINK_IMG_W, cfg.TRAIN.SHRINK_IMG_H), interpolation=cv2.INTER_CUBIC)

        img = img.astype('float32')  # 转换类型为float32
        img -= np.mean(img)
        img /= 255.
        # print(img.shape)
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


