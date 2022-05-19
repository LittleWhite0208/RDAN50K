import torch
import os
import logging
from collections import defaultdict, Iterable
from copy import deepcopy
from itertools import chain

from core.config import cfg
import numpy as np

logger = logging.getLogger(__name__)

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

class AUCMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.score_labels = []
        self.pos_count = 0
        self.neg_count = 0

    def update(self, pos_scores, target):
        target = target.cpu().numpy()
        self.pos_count += np.sum(target == 1)
        self.neg_count += np.sum(target == 0)
        n = target.shape[0]
        for i in range(n):
            self.score_labels.append((pos_scores[i], target[i]))

    def get_auc(self):
        if self.pos_count == 0 or self.neg_count == 0:
            return 0
        self.score_labels.sort()
        pos_rank_sum = 0
        for (i, (score, label)) in enumerate(self.score_labels):
            if label == 1:
                pos_rank_sum += (i+1)
        return 100.0 * (pos_rank_sum - (self.pos_count * (self.pos_count + 1)) * 0.5) / (self.pos_count * self.neg_count)

def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio

def update_learning_rate(optimizer, cur_lr, new_lr):
    """Update learning rate"""
    if cur_lr != new_lr:
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
            logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
        # Update learning rate, note that different parameter may have different learning rate
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            if ind == 1 and cfg.MODEL.REC:  # bias params
                param_group['lr'] = new_lr * cfg.TRAIN.LR_RATIO#2
            else:
                param_group['lr'] = new_lr
            param_keys += param_group['params']
        if cfg.SOLVER.TYPE in ['SGD'] and cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            _CorrectMomentum(optimizer, param_keys, new_lr / cur_lr)

def adjust_learning_rate(base_lr, step_values, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr
    for step_value in step_values:
        if epoch >= step_value:
            lr *= 0.1
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_ckpt(output_dir, args, epoch, step, model, optimizer, iters_per_epoch):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_{}_{}.pth'.format(epoch, step))
    torch.save({
        'epoch': epoch,
        'step': step,
        'iters_per_epoch': iters_per_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)

def train_save_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)

def load_ckpt(model, ckpt):
    """Load checkpoint"""
    state_dict = {}
    for name in ckpt:
        state_dict[name] = ckpt[name]
    model.load_state_dict(state_dict, strict=False)

def load_optimizer_state_dict(optimizer, state_dict):
    # deepcopy, to be consistent with module API
    state_dict = deepcopy(state_dict)
    # Validate the state_dict
    groups = optimizer.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of "
                         "parameter groups")
    param_lens = (len(g['params']) for g in groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Update the state
    id_map = {old_id: p for old_id, p in
                zip(chain(*(g['params'] for g in saved_groups)),
                    chain(*(g['params'] for g in groups)))}

    def cast(param, value):
        """Make a deep copy of value, casting all tensors to device of param."""
        if torch.is_tensor(value):
            # Floating-point types are a bit special here. They are the only ones
            # that are assumed to always match the type of params.
            if isinstance(param.data, (torch.FloatTensor, torch.cuda.FloatTensor,
                                       torch.DoubleTensor, torch.cuda.DoubleTensor,
                                       torch.HalfTensor, torch.cuda.HalfTensor)):  # param.is_floating_point():
                value = value.type_as(param.data)
            value = value.cuda(param.get_device()) if param.is_cuda else value.cpu()
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v) for k, v in value.items()}
        elif isinstance(value, Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state = defaultdict(dict)
    for k, v in state_dict['state'].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group, new_group):
        new_group['params'] = group['params']
        return new_group
    param_groups = [
        update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    optimizer.__setstate__({'state': state, 'param_groups': param_groups})

def _CorrectMomentum(optimizer, param_keys, correction):
    """The MomentumSGDUpdate op implements the update V as

        V := mu * V + lr * grad,

    where mu is the momentum factor, lr is the learning rate, and grad is
    the stochastic gradient. Since V is not defined independently of the
    learning rate (as it should ideally be), when the learning rate is
    changed we should scale the update history V in order to make it
    compatible in scale with lr * grad.
    """
    logger.info('Scaling update history by %.6f (new lr / old lr)', correction)
    for p_key in param_keys:
        optimizer.state[p_key]['momentum_buffer'] *= correction

def decay_learning_rate(optimizer, cur_lr, decay_rate):
    """Decay learning rate"""
    new_lr = cur_lr * decay_rate
    # ratio = _get_lr_change_ratio(cur_lr, new_lr)
    ratio = 1 / decay_rate
    if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
        logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
    # Update learning rate, note that different parameter may have different learning rate
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        new_lr = decay_rate * param_group['lr']
        param_group['lr'] = new_lr
        if cfg.SOLVER.TYPE in ['SGD']:
            if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                    ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
                _CorrectMomentum(optimizer, param_group['params'], new_lr / cur_lr)
