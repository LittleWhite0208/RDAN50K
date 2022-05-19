#-*- coding: UTF-8 -*- 

import argparse
import os
import logging
import random
import shutil
import time
import warnings
import sys
import re

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from myLog import log_info


import _init_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


smooth = 0.01
threshold = 0.5


from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from utils.mio import load_string_list
from utils.image_utils import rescale_img, label_extract, label_to_array
from datasets.betaroad30_190311 import BetaroadDataset30, my_collate_fn
# from models.densenet import DenseNet,densenet121
from models.densenet_res_CBAM import densenet121,PSDiceLoss
from models.resnet34 import resnet18, resnet34, PSDiceLoss
from models.resnet50 import resnet50
from models.resnet101 import resnet101
from models.replknet import create_RepLKNet31B

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a Rec_Seg network')
    parser.add_argument(
        '--cfg', dest='cfg_file',  default='../configs/dist.yml',
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')
    parser.add_argument(
        '--print_freq',
        help='Display training info every N iterations',
        default=10, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')
    parser.add_argument(
        '--bs', dest='batch_size', default=1,
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers', default=4,
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)
    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)

    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_epochs',
        help='Epochs to decay the learning rate on. '
             'Decay happens on the beginning of a epoch. '
             'Epoch is 0-indexed.',
        default=[2, 6], nargs='+', type=int)


    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default=None, type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default=None, type=str,
                        help='distributed backend')
    parser.add_argument('--save_modelpath', default=None, type=str,
                        help='the model path of pkl')

    # Epoch
    parser.add_argument(
        '--start_iter',
        help='Starting iteration for first training epoch. 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--start_epoch',
        help='Starting epoch count. Epoch is 0-indexed.',
        default=0, type=int)

    parser.add_argument(
        '--epochs', dest='num_epochs',
        help='Number of epochs to train',
        default=1000, type=int)


    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
        metavar='LR', help='initial learning rate', dest='lr')

    # Checkpoint and Logging
    parser.add_argument(
        '--output_base_dir',
        help='Output base directory',
        default="Outputs")

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--ckpt_num_per_epoch',
        help='number of checkpoints to save in each epoch. '
             'Not include the one at the end of an epoch.',
        default=1, type=int)

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true', default=False)

    return parser.parse_args()




def main():
    """Main function"""
    args = parse_args()
    print('Called with args:')

    cfg_from_file(args.cfg_file)
    if args.rank > 0:
        cfg.rank = args.rank
    else:
        args.rank = cfg.rank

    if args.world_size > 0:
        cfg.world_size = args.world_size
    else:
        args.world_size = cfg.world_size


    if args.dist_url is not None:
        cfg.dist_url = args.dist_url
    else:
        args.dist_url = cfg.dist_url


    if args.dist_backend is not None:
        cfg.dist_backend = args.dist_backend
    else:
        args.dist_backend = cfg.dist_backend

    if args.save_modelpath is not None:
        cfg.save_modelpath = args.save_modelpath
    else:
        args.save_modelpath = cfg.save_modelpath


    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if not torch.cuda.is_available():
        raise ValueError("Need Cuda device to run !")

    ngpus_per_node = torch.cuda.device_count()
    print('rank=%d, gpu number=%d.'%(args.rank, ngpus_per_node))
    main_worker(None, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    best_acc1 = 0

    print(args)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    # create model
    model = create_RepLKNet31B(small_kernel_merged=False)
    #model = Net()
    model.cuda()
    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    model = torch.nn.parallel.DistributedDataParallel(model)
    #加载模型

    modelpath=args.save_modelpath

    if (os.path.exists(modelpath) == False):
           os.makedirs(modelpath)
    bestModel,args.start_epoch =find_lastModel(modelpath)
    print("model depend:",bestModel)
    print("start_epoch:", args.start_epoch)
       # 加载模型
    if bestModel is not None and os.path.exists(bestModel):
        model.load_state_dict(torch.load(bestModel))


    # define loss function (criterion) and optimizer
    criterion = PSDiceLoss(smooth, threshold).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    g_time = time.time()
    train_dataset = BetaroadDataset30("CRACK",cfg.TASK.TRAIN_FILE)
    val_dataset = BetaroadDataset30("CRACK",cfg.TASK.VAL_FILE)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader( train_dataset, collate_fn=my_collate_fn, 
        batch_size=cfg.TASK.BATCHSIZE, num_workers=10, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader( val_dataset, collate_fn=my_collate_fn, 
        batch_size=cfg.TASK.BATCHSIZE, num_workers=10, pin_memory=True)

     
    for epoch in range(args.start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_index=train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        #保存模型：
        if args.rank % ngpus_per_node == 0:
            pkl_name =get_pklname(args.save_modelpath,epoch + 1, int(10000 * train_index), int(10000 * acc1))
            torch.save(model.state_dict(), pkl_name)

        #将save_checkpoint给注释掉，不需要
        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)
        #
        # if args.rank % ngpus_per_node == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': 0, #args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best)

def find_lastModel(path):
        # 在指定路径下找到最后的模型，返回模型的文件名。
        modelList = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if re.search("pkl", file):
                    indexName = os.path.join(root, file)
                    # print(indexName)
                    modelList.append(indexName)
        print(len(modelList))
        if (len(modelList) == 0):
            return None,0

        lastmodel = max(modelList)
        print(lastmodel)
        return lastmodel,len(modelList)



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        #print("inpusize:",np.shape(input))
        target = target.cuda(non_blocking=True)
        # print("np.shape(input):",np.shape(input))
        # print("np.shape(target):", np.shape(target))

        # compute output
        output = model(input)
        print("output:",output.shape)
        loss = criterion(output, target)
        #print("loss:",loss)

        # measure accuracy and record loss
        acc1 = accuracy(output, target)
        #print("acc1:",acc1)
        losses.update(loss.item(), input.size(0))
        #print("loss.item():",loss.item())
        #print("input.size(0):",input.size(0))
        top1.update(acc1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
    return top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def get_pklname(modelpath,epoch, train_index, val_index):
    pklpath =modelpath
    trainval_info = '-%03d-%04d-%04d.pkl' % (epoch, train_index, val_index)
    name =os.path.basename(modelpath) + trainval_info
    pkl_name = os.path.join(pklpath, name)
    print("pkl_name:", pkl_name)
    return pkl_name


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target):
    y_pred_t = torch.gt(output,threshold).float()
    y_true_t = torch.gt(target,threshold).float()
    intersection = torch.sum(y_pred_t*y_true_t) ##注意* 和 &的差异
    dice = (2.0*intersection+smooth)/(torch.sum(y_true_t)+torch.sum(y_pred_t)+smooth)
    return dice.item()

if __name__ == '__main__':
    main()
