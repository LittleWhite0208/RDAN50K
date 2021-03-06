import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
 
__C = edict()
# Consumers can get config by:
# 在其他文件使用config要加的命令，例子见train_net.py
#   from fast_rcnn_config import cfg
cfg = __C
 
#
# Training options
# 训练的选项
#
 
__C.TRAIN = edict()
 
# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
# 最短边Scale成600
__C.TRAIN.SCALES = (600,)
 
# Max pixel size of the longest side of a scaled input image
# 最长边最大为1000
__C.TRAIN.MAX_SIZE = 1000
 
# Images to use per minibatch
# 一个minibatch包含两张图片
__C.TRAIN.IMS_PER_BATCH = 2
 
# Minibatch size (number of regions of interest [ROIs])
#  Minibatch大小，即ROI的数量
__C.TRAIN.BATCH_SIZE = 128
 
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
# minibatch中前景样本所占的比例
__C.TRAIN.FG_FRACTION = 0.25
 
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
# 与前景的overlap大于等于0.5认为该ROI为前景样本
__C.TRAIN.FG_THRESH = 0.5
 
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
# 与前景的overlap在0.1-0.5认为该ROI为背景样本
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1
 
# Use horizontally-flipped images during training?
# 水平翻转图像，增加数据量
__C.TRAIN.USE_FLIPPED = True
 
# Train bounding-box regressors
# 训练bb回归器
__C.TRAIN.BBOX_REG = True
 
# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
# BBOX阈值，只有ROI与gt的重叠度大于阈值，这样的ROI才能用作bb回归的训练样本
__C.TRAIN.BBOX_THRESH = 0.5
 
# Iterations between snapshots
# 每迭代1000次产生一次snapshot
__C.TRAIN.SNAPSHOT_ITERS = 10000
 
# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
# 为产生的snapshot文件名称添加一个可选的infix. solver.prototxt指定了snapshot名称的前缀
__C.TRAIN.SNAPSHOT_INFIX = ''
 
# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
# 在roi_data_layer.layer使用预取线程，作者认为不太有效，因此设为False
__C.TRAIN.USE_PREFETCH = False
 
# Normalize the targets (subtract empirical mean, divide by empirical stddev)
# 归一化目标BBOX_NORMALIZE_TARGETS，减去经验均值，除以标准差
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
# 弃用
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
# 在BBOX_NORMALIZE_TARGETS为True时，归一化targets,使用经验均值和方差
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
 
# Train using these proposals
# 使用'selective_search'的proposal训练！注意该文件来自fast rcnn，下文提到RPN
__C.TRAIN.PROPOSAL_METHOD = 'selective_search'
 
# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
# minibatch的两个图片应该有相似的宽高比，以避免冗余的zero-padding计算
__C.TRAIN.ASPECT_GROUPING = True
 
# Use RPN to detect objects
# 使用RPN检测目标
__C.TRAIN.HAS_RPN = False
# IOU >= thresh: positive example
# RPN的正样本阈值
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
# RPN的负样本阈值
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
# 如果一个anchor同时满足正负样本条件，设为负样本（应该用不到）
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
# 前景样本的比例
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
# batch size大小
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
# 非极大值抑制的阈值
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
# 在对RPN proposal使用NMS前，要保留的top scores的box数量
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
# 在对RPN proposal使用NMS后，要保留的top scores的box数量
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# proposal的高和宽都应该大于RPN_MIN_SIZE，否则，映射到conv5上不足一个像素点
__C.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
# 弃用
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# 给定正RPN样本的权重
# and give negatives a weight of (1 - p)
# 给定负RPN样本的权重
# Set to -1.0 to use uniform example weighting
# 这里正负样本使用相同权重
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
 
#
# Testing options
# 测试选项
#
 
__C.TEST = edict()
 
# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)
 
# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000
 
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
# 测试时非极大值抑制的阈值
__C.TEST.NMS = 0.3
 
# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
# 分类不再用SVM，设置为False
__C.TEST.SVM = False
 
# Test using bounding-box regressors
# 使用bb回归
__C.TEST.BBOX_REG = True
 
# Propose boxes
# 不使用RPN生成proposal
__C.TEST.HAS_RPN = False
 
# Test using these proposals
# 使用selective_search生成proposal
__C.TEST.PROPOSAL_METHOD = 'selective_search'
 
## NMS threshold used on RPN proposals
#  RPN proposal的NMS阈值
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16
 
#
# MISC
#
 
# The mapping from image coordinates to feature map coordinates might cause
# 从原图到feature map的坐标映射，可能会造成在原图上不同的box到了feature map坐标系上变得相同了
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
# 缩放因子
__C.DEDUP_BOXES = 1./16.
 
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# 所有network所用的像素均值设为相同
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
 
# For reproducibility
__C.RNG_SEED = 3
 
# A small number that's used many times
# 极小的数
__C.EPS = 1e-14
 
# Root directory of project
# 项目根路径
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
 
# Data directory
# 数据路径
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
 
# Model directory
# 模型路径
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'pascal_voc'))
 
# Name (or path to) the matlab executable
# matlab executable
__C.MATLAB = 'matlab'
 
# Place outputs under an experiments directory
# 输出在experiments路径下
__C.EXP_DIR = 'default'
 
# Use GPU implementation of non-maximum suppression
# GPU实施非极大值抑制
__C.USE_GPU_NMS = True
 
# Default GPU device id
# 默认GPU id
__C.GPU_ID = 0
 
def get_output_dir(imdb, net=None):
    #返回输出路径，在experiments路径下
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
 
    A canonical标准 path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
 
def _merge_a_into_b(a, b):
    #两个配置文件融合
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return
 
    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))
 
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))
 
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        #用配置a更新配置b的对应项
        else:
            b[k] = v
 
def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    # 导入配置文件并与默认选项融合
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
 
    _merge_a_into_b(yaml_cfg, __C)
 
def cfg_from_list(cfg_list):
    # 命令行设置config
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value