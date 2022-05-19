#-*- coding: UTF-8 -*- 
from __future__ import print_function
from numpy.lib import pad
import os
import time
import queue
import cv2
import threading
import inspect
import ctypes
import random
import numpy as np

import torch
from core.config import cfg
from datasets.betaroad30 import BetaroadDataset30, my_collate_fn

'''''
这个地方就是要求分成多少个机器
然后每一个机器看看有多少个gpu，一个gpu一个
进程，故事就是这样开始的
就是确定拿一台机器
'''
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """
    def __init__(self, btz = None):
        self.plan = cfg.TASK.PLAN
        self.train_datafile = cfg.TASK.TRAIN_FILE
        self.val_datafile = cfg.TASK.VAL_FILE
        if btz is None:
            self.btz = cfg.TASK.BATCHSIZE
        else:
            self.btz = btz
        self.check_ratio()

    def check_ratio(self):
        ratio = 0
        for item in self.plan:
            ratio += item[1]
        if ratio != 1:
            raise ValueError("ratio sum must be 1!")

    def __len__(self):
        return len(self.plan)

    '''返回某一个机器的训练数据，[[0,0.3],[1,0.7]]，总训练数据据量10000条
        总验证数据2000条，
        例如：第1号机器，占0.7，有4个gpu
        那么，该函数反回如下列表：
        (2,10, (3000,4749),(600,949))
        分别表示：rank，该机器的batch_size, 训练开始行号和结束行号，验证开始行号和结束行号
        no是0-based
    '''
    def __getitem__(self, no):
        if no >= len(self.plan):
            raise ValueError("no. must is in plan!")
        return self.getRanges(no)

    def getRanges(self, no):
        btz = self.btz
        print(self.plan[no])
        train_lines = self.getLines(self.train_datafile)
        val_lines = self.getLines(self.val_datafile)

        print('Gen-TRAIN_NUM:%d,VAL_NUM:%d.'%(train_lines,val_lines))
        ratio_prev = 0
        ratio = self.plan[no][1]
        btz = int(round(btz * ratio))
        for i in range(no):
            ratio_prev += self.plan[i][1] 
            
        train_steps = int(round(ratio * train_lines))
        val_steps = int(round(ratio * val_lines))

        print('train_steps is %d, val_step is %d, ratio_prev is %d'%(train_steps, val_steps, ratio_prev))

        train_start = int(ratio_prev * train_lines) 
        train_end = train_start + train_steps - 1       
  
        print('train::start is %d, end is %d.'%(train_start, train_end))
        
        val_start = int(round(ratio_prev * val_lines))
        val_end = val_start + val_steps - 1 
 
        print('val::start is %d, end is %d.'%(val_start, val_end))

        rank = no
        return (rank, btz, (train_start, train_end), (val_start, val_end))      

    def getLines(self, datafile):
        count = 0
        try:
            f = open(datafile, "r")
            for line in f:
                count += 1
            f.close()
        except IOError as err:
            print('getLines:File error:'+str(err))
        return count


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, no, btz=None):
        self.plan = cfg.TASK.PLAN
        self.no = no
        if btz is None:
            self.btz = cfg.TASK.BATCHSIZE
        else:
            self.btz = btz
        self.partitions = Partition(self.btz)

    def use(self):
        self.rgs = self.partitions[self.no]
        print(self.rgs)
        # rank,batch_size, train_range, val_range
        train_tuple = (cfg.TASK.TRAIN_FILE, self.rgs[2][0], self.rgs[2][1])
        val_tuple = (cfg.TASK.VAL_FILE, self.rgs[3][0], self.rgs[3][1])

        #print(train_tuple)
        #print(val_tuple)

        train_dataset = BetaroadDataset30('train', train_tuple)
        val_dataset = BetaroadDataset30('val', val_tuple)

        train_set = torch.utils.data.DataLoader(
            train_dataset, num_workers=20, collate_fn=my_collate_fn, batch_size=self.rgs[1])

        val_set = torch.utils.data.DataLoader(
            val_dataset, num_workers=20, collate_fn=my_collate_fn, batch_size=self.rgs[1])

        # rank = no
        # return transet, valset, rank, this mathine batchsize
        return (train_set, val_set, self.rgs[0], self.rgs[1])



