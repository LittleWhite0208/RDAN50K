#-*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import copy
from ast import literal_eval

import numpy as np
import yaml

from utils.mycollections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


# ---------------------------------------------------------------------------- #
# Patch options
# ---------------------------------------------------------------------------- #
__C.PATCH = AttrDict()
__C.PATCH.BLOCK_SIZE = 50
__C.PATCH.STRIDE = 32
__C.PATCH.PAD = 25
__C.PATCH.SIZE = 256
__C.PATCH.FILTER = True
__C.PATCH.FILTER_SIZE = 51
__C.PATCH.BLOCK_NUM = 8

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #
__C.DATA = AttrDict()
__C.DATA.PADDING = 0
__C.DATA.AUG = False

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# Number of classes in the dataset; must be set
# E.g., 81 for COCO (80 foreground + 1 background)
__C.MODEL.NUM_CLASSES = -1
__C.MODEL.REC = False
__C.MODEL.IDEA = '0'

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

__C.TRAIN.IMG_BLOCK_H = 22
__C.TRAIN.IMG_BLOCK_W = 34
__C.TRAIN.RAW_IMAGE_ROWS = 2200
__C.TRAIN.RAW_IMAGE_COLS = 3400
__C.TRAIN.SHRINK_IMG_W = 704
__C.TRAIN.SHRINK_IMG_H = 1088
__C.TRAIN.SNAPSHOT_ITERS = 2000
__C.TRAIN.EVAL = True


__C.TRAIN.DATASETS = ()

__C.TRAIN.IMS_PER_BATCH = 128
__C.TRAIN.RAW_BLOCK_SIZE = 50
__C.TRAIN.BlOCK_HEIGHT = 41
__C.TRAIN.BLOCK_WIDTH = 62
__C.TRAIN.NUM_FILTERS = (32, 64, 128)
__C.TRAIN.NUM_BLOCKS = (0, 0, 3)
__C.TRAIN.CONV = 2
__C.TRAIN.DILATION = False
__C.TRAIN.NORM = 'l1'
__C.TRAIN.CRITERION = 'bce' # 'dice'

__C.TRAIN.SNAPSHOT_ITERS = 20000
__C.TRAIN.EVAL = False
__C.TRAIN.REC_WEIGHT = 0.5
__C.TRAIN.LR_RATIO = 1.

__C.TRAIN.LOSS = '0'


# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = AttrDict()

# Number of Python threads to use for the data loader (warning: using too many
# threads can cause GIL-based interference with Python Ops leading to *slower*
# training; 4 seems to be the sweet spot in our experience)
__C.DATA_LOADER.NUM_THREADS = 4

# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# e.g 'SGD', 'Adam'
__C.SOLVER.TYPE = 'SGD'

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = 'step'

# Hyperparameter used by the specified policy
# For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Uniform step size for 'steps' policy
__C.SOLVER.STEP_SIZE = 30000
__C.SOLVER.STEPS = []
__C.SOLVER.LRS = []
__C.SOLVER.MAX_ITER = 40000

__C.SOLVER.MOMENTUM = 0.9
# lr_step_values = [15, 25, 28]
__C.SOLVER.WEIGHT_DECAY = 0.0005
# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500
# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0

# WARM_UP_METHOD can be either 'constant' or 'linear' (i.e., gradual)
__C.SOLVER.WARM_UP_METHOD = 'linear'

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1

# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()
# Datasets to test on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, testing is performed on each one sequentially
__C.TEST.DATASETS = ()


__C.LABEL_TYPE = AttrDict()
__C.LABEL_TYPE.CRACK = 'Bad_BlockPos'
__C.LABEL_TYPE.REPAIR = 'Bad_RepairBlockPos'

__C.TASK = AttrDict()
__C.TASK.TRAIN_FILE = '/home/xgs/train.txt'
__C.TASK.VAL_FILE = '/home/xgs/val.txt'
__C.TASK.PLAN = [[0,0.3,2],[1,0.5,4],[1,0.2,4]]
__C.TASK.BATCHSIZE = 50


# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# [Infered value]
__C.CUDA = False
__C.DEBUG = False
__C.dist_backend = 'nccl'
__C.dist_url = 'tcp://224.66.41.62:23456'
__C.rank = 0
__C.world_size = 3
__C.lr = 0.1
__C.save_modelpath = './pklmodel'




def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value
def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a

def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v

cfg_from_list = merge_cfg_from_list

def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if make_immutable:
        cfg.immutable(True)

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v

def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file
