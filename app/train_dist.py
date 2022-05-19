#-*- coding: UTF-8 -*-
# library
# standard library
import os
import numpy as np
import sys
from graphviz import Digraph 

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import DataParallel
import torch
from core.config import cfg
from datasets.betaroad30 import BetaroadDataset30, my_collate_fn



from datasets.betaroad30 import BetaroadDataset30



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



class train_2018001():
    def __init__(self, cfg_ctx):
        self.cfg_ctx = cfg_ctx
        self.smooth = 1
        self.threshold = 0.5
        self.h = self.cfg_ctx['SHRINK_IMG_HEIGHT']
        self.w = self.cfg_ctx['SHRINK_IMG_WIDTH']
        self.is_have_model = True
        self.cnn = DataParallel(net_2018_001())
        self.cnn.cuda()

    def average_gradients(self, model):
        """ Gradient averaging. """
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            param.grad.data /= size

    def count_accuracy(self, pred_y, test_y):
        #pred_y[pred_y >  self.threshold] = 1
        #pred_y[pred_y <= self.threshold] = 0
        tp = ((pred_y.data > self.threshold) & (test_y.data > self.threshold)).sum()
        tn = ((pred_y.data <=self.threshold) & (test_y.data <=self.threshold)).sum()
        fp = ((pred_y.data > self.threshold) & (test_y.data <=self.threshold)).sum()
        fn = ((pred_y.data <=self.threshold) & (test_y.data > self.threshold)).sum()
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        #f1 = 2*r*p/(r+p)
        acc = (tp+tn)/(tp+tn+fp+fn) #不管正负样本，凡是对的，不管正样本对，还是负样本对都认可
        return acc
    '''
    def count_dice(self, pred_y, test_y):
        #tp = ((pred_y.data > self.threshold) & (test_y.data > self.threshold )).sum()
        pred_y[pred_y > 0.5] = 1
        pred_y[pred_y <= 0.5] = 0
        #tp = (pred_y.data & test_y.data).sum()
        tp = (pred_y.int().data & test_y.int().data).sum() ## &符号要求整型，但是*可以不要求，经过测试，效果一样
        pred_sum = pred_y.data.sum()
        test_sum = test_y.data.sum()
        dice = (2.0*tp+self.smooth)/(pred_sum+test_sum+self.smooth)
        return dice
    '''

    def count_dice(self, y_pred, y_true):
        y_pred_t = torch.gt(y_pred,self.threshold).float()
        y_true_t = torch.gt(y_true,self.threshold).float()
        intersection = torch.sum(y_pred_t*y_true_t) ##注意* 和 &的差异
        dice = (2.0*intersection+self.smooth)/(torch.sum(y_true_t)+torch.sum(y_pred_t)+self.smooth)
        return dice.data[0]

    #  这个gpu的数据的batch_size，不是全部数据的
    def train(self, train_set, val_set, batch_size):
        if os.path.exists(self.cfg_ctx['MODEL_NAME_DEPEND']):
            self.cnn.load_state_dict(torch.load(self.cfg_ctx['MODEL_NAME_DEPEND']))

        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=1e-4)
        #loss_func = nn.CrossEntropyLoss()
        #loss_func = nn.BCELoss()
        #loss_func = nn.MSELoss()
        loss_func = PSDiceLoss(self.smooth, self.threshold)

        epochs=1000

        ##print('\r\n')
        sys.stdout.flush()
        max_ind = 0
        ind = []
        los_list = []
        for epoch in range(epochs):
            ind.clear()
            self.cnn.train()

            for i,X,Y  in train_set:
                X = torch.from_numpy(X)
                x = Variable(X).cuda()

                Y = torch.from_numpy(Y).float()
                y = Variable(Y).cuda()

                #print(x.data.size())
                fx = self.cnn(x)
                #index = self.count_accuracy(fx, y)
                index = self.count_dice(fx, y)

                ind.append(index)
                index_cur = sum(ind) / float(len(ind))

                #print(fx)
                #print(fx.data.size())
                #print(y.data.size())
                #loss = dice_loss(fx, y, self.threshold, self.smooth)         # cross entropy loss
                loss = loss_func(fx, y)         # cross entropy loss
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                self.average_gradients(self.cnn)
                optimizer.step()                # apply gradients

                los_list.append(loss.data[0])
                los_cur = sum(los_list) / float(len(los_list))
                #print('Epoch: ', i, '| train loss: %.4f' % loss.data[0], end='\r')
                #print('TRAIN!!!Epoch: %d/%d | batch: %d/%d | train index: %.4f | train loss: %.4f' % (i+1, epochs, k+1, steps_of_epoch, index_cur, loss.data[0]), end='\r')
                print('\rTRAIN!!!Epoch: %d/%d | batch: %d/%d | train index: %.4f | train loss: %.4f' % (i+1, epochs, k+1, steps_of_epoch, index_cur, los_cur), end='')
                sys.stdout.flush() 

            print('\r\n',end='')
            self.cnn.eval()
            ind.clear()
            for X,Y  in train_set:
                X = torch.from_numpy(X)
                x = Variable(X).cuda()

                Y = torch.from_numpy(Y).float()
                y = Variable(Y).cuda()

                fx = self.cnn(x)
                #print(fx.data.size())
                #index = self.count_accuracy(fx, y)
                index = self.count_dice(fx, y)
                ind.append(index)
                index_cur = sum(ind) / float(len(ind))
                
                print('\rVAL!!!Epoch: %d/%d | batch: %d/%d | test index: %.4f' % (i+1, epochs, k+1, steps_of_epoch_val, index_cur), end='')
                sys.stdout.flush() 

            print('\r\n',end='')
            index_epoch = sum(ind) / float(len(ind))
            print('VAL!!!Epoch: %d/%d | average index/max index: %.4f/%.4f ' % (i+1, epochs, index_epoch, max_ind))
            sys.stdout.flush() 
            if index_epoch > max_ind:
                max_ind = index_epoch
                pkl_name = self.cfg_ctx['MODEL_NAME'] % (i+1, int(10000*max_ind))
                print('VAL!!!save pkl to filename=%s' % pkl_name)
                sys.stdout.flush() 
                torch.save(self.cnn.state_dict(), pkl_name)   # save only the parameters


    def load_weight(self, mode_type, model_filename_type):
        model_filename = self.cfg_ctx[model_filename_type]
        if model_filename is None:
            print('[%s]模型文件未定义，程序不会进行该类型的识别！' % mode_type)
            sys.stdout.flush() 
            self.is_have_model = False
            return False
        if os.path.exists(model_filename):
            self.cnn.load_state_dict(torch.load(model_filename_type))
            return True
        else:
            print('[%s]模型文件不存在，程序不会进行该类型的识别！' % mode_type)
            self.is_have_model = False
            return False


    def run(rank, size):
        """ Distributed Synchronous SGD Example """
        torch.manual_seed(1234)
        train_set, bsz = partition_dataset()
        model = Net()
        model = model
    #    model = model.cuda(rank)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        num_batches = ceil(len(train_set.dataset) / float(bsz))
        for epoch in range(10):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = Variable(data), Variable(target)
    #            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                epoch_loss += loss.data[0]
                loss.backward()
                average_gradients(model)
                optimizer.step()
            print('Rank ',
                  dist.get_rank(), ', epoch ', epoch, ': ',
                  epoch_loss / num_batches)


    def init_processes(rank, size, fn, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)

    ### 这是第no台机器
    def main(self, no):
        processes = []
        for num in DataPartitioner(no):
            p = Process(target=init_processes, args=(rank, size, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

