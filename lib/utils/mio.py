#-*- coding:utf-8 -*-
import os, time
import torch
import numpy as np

def save_data_list(x, y, save_dir):
    from sklearn.model_selection import train_test_split
    train, valid, train_y, valid_y = train_test_split(
        x, y, stratify=y, test_size=0.15, random_state=12345)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_string_list(save_dir + 'valid_name_list.txt', valid)
    save_string_list(save_dir + 'train_name_list.txt', train)

def mkdir_safe(d):
    sub_dirs = d.split('/')
    cur_dir = ''
    max_check_time = 5
    sleep_seconds_per_check = 0.001
    for i in range(len(sub_dirs)):
        cur_dir += sub_dirs[i] + '/'
        for check_iter in range(max_check_time):
            if not os.path.exists(cur_dir):
                try:
                    os.mkdir(cur_dir)
                except Exception as e:
                    print('[WARNING] ', str(e))
                    time.sleep(sleep_seconds_per_check)
                    continue
            else:
                break


def load_string_list(file_path):
    try:
        f = open(file_path)
        l = []
        for item in f:
            item = item.strip()
            if len(item) == 0:
                continue
            l.append(item)
        f.close()
    except IOError:
        print('open error', file_path)
        return None
    else:
        return l

def save_string_list(file_path, l):
    f = open(file_path, 'w')
    for item in l[:-1]:
        f.write(item + '\n')
    if len(l) >= 1:
        f.write(l[-1])
    f.close()

def create_log_config(save_path):
    lines = load_string_list('/home/luoling/multi/logging.conf')
    new_lines = []
    for line in lines:
        new_lines.append(line.replace('./test.log', save_path))
    save_dir = os.path.dirname(save_path)
    mkdir_safe(save_dir)
    save_string_list(save_dir + '/logging.conf', new_lines)
    return save_dir + '/logging.conf'

