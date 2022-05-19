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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (x, 1, 704, 1088)
            nn.BatchNorm2d(1),
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        # self.conv1 = nn.DataParallel(self.conv1)

        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.Conv2d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        # self.conv2 = nn.DataParallel(self.conv2)

        self.conv3 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,  # input height
                out_channels=128,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.Conv2d(
                in_channels=128,  # input height
                out_channels=128,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )

        self.conv4 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,  # input height
                out_channels=256,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )

        self.conv5 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,  # input height
                out_channels=256,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )

        self.conv6 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,  # input height
                out_channels=512,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
        )

        self.conv7 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(512),
            nn.Conv2d(
                in_channels=512,  # input height
                out_channels=512,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
        )
        self.conv8 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(512),
            nn.Conv2d(
                in_channels=512,  # input height
                out_channels=1024,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
        )
        self.conv9 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(1024),
            nn.Conv2d(
                in_channels=1024,  # input height
                out_channels=1024,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
        )
        self.conv10 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(1024),
            nn.Conv2d(
                in_channels=1024,  # input height
                out_channels=2048,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
        )

        self.conv11 = nn.Sequential(  # input shape (1, 28, 28)
            nn.BatchNorm2d(2048),
            nn.Conv2d(
                in_channels=2048,  # input height
                out_channels=1,  # n_filters
                kernel_size=1,  # filter size
                stride=1,  # filter movement/step
            ),  # output shape (16, 28, 28)
            nn.Sigmoid(),  # activation
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        output = self.conv11(x)
        return output

def dice_loss2(y_pred, y_true, threshold, smooth = 1):
    tp = ((y_pred.data > threshold) & (y_true.data > threshold )).sum()
    pred_sum = y_pred.data.sum()
    true_sum = y_true.data.sum()
    dice_tensor = torch.FloatTensor(1).zero_()
    dice_tensor[0] = 1.0-(2.0*tp+smooth)/(pred_sum+true_sum+smooth)
    return Variable(dice_tensor,requires_grad=True).cuda()

def dice_loss(y_pred, y_true, threshold, smooth = 1):
    y_pred[y_pred >  threshold] = 1
    y_pred[y_pred <= threshold] = 0
    intersection = torch.sum(y_true*y_pred) ##注意* 和 &的差异
    return 1.0-(2.*intersection+smooth)/(torch.sum(y_true)+torch.sum(y_pred)+smooth)
'''
def dice_loss(y_pred, y_true, threshold, smooth = 1):
    #y_pred[y_pred >  threshold] = 1
    #y_pred[y_pred <= threshold] = 0
    intersection = torch.sum(y_true * y_pred) ##注意* 和 &的差异
    return 1.0-(2.*intersection+smooth)/(torch.sum(y_true)+torch.sum(y_pred)+smooth)
'''

class PSDiceLoss(nn.Module):
    def __init__(self, smooth, threshold):
        super(PSDiceLoss, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
    '''
    def forward(self, pred_y, label_y):
        tp = torch.sum(torch.gt(pred_y, self.threshold)*torch.gt(label_y, self.threshold)).float()
        #tp = ((pred_y.data > self.threshold) & (label_y.data > self.threshold )).sum()
        pred_sum = torch.sum(torch.gt(pred_y, self.threshold)).float() #pred_y.data.sum()
        label_sum = label_y.data.sum()
        dice_tensor = torch.FloatTensor(1).zero_()
        dice_tensor[0] = 1.0-(2.0*tp+self.smooth)/(pred_sum+label_sum+self.smooth)
        return Variable(dice_tensor,requires_grad=True).cuda()
    '''

    def forward(self, pred_y, label_y):
        label_y = label_y.type(torch.cuda.FloatTensor) # 转Float
        intersection = torch.sum(pred_y*label_y) ##注意* 和 &的差异
        return 1.0-(2.*intersection+self.smooth)/(torch.sum(pred_y)+torch.sum(label_y)+self.smooth)

    '''
    def forward(self, pred_y, label_y):
        tp = torch.sum(torch.gt(pred_y, self.threshold)*torch.gt(label_y, self.threshold)).float()
        pred_sum = torch.sum(torch.gt(pred_y, self.threshold)).float()
        label_sum = torch.sum(torch.gt(label_y, self.threshold)).float()
        div_sum = torch.add(pred_sum, label_sum)
        div_sum = torch.add(div_sum, self.smooth)
        tp = torch.mul(tp, 2.0)
        tp = torch.add(tp, self.smooth)
        los = torch.div(tp, div_sum)
        los = torch.mul(los, -1.0)
        los = torch.add(los, 1.0)

        ##_losses = 1.0-(2.0*tp+self.smooth)/(pred_sum+label_sum+self.smooth)
        print(los)
        return los
    '''
