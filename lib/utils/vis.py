#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from core.config import cfg


def hw_extract(filepath):
    lines = open(filepath)
    height = 0
    width = 0
    for line in lines:
        if 'PicSrcHeight' in line:
            height = line.split(':')[-1].strip('\r\n')
        if 'PicSrcWidth' in line:
            width = line.split(':')[-1].strip('\r\n')
            break
    return height, width

# 提取label坐标
def label_extract(filepath, label_type=cfg.LABEL_TYPE):
    label_list = []
    lines = open(filepath)
    for line in lines:
        if line.find(label_type.encode('utf-8')) != -1:
            ax = line.strip("\n").replace(label_type+':', "").split(",")
            ax = tuple( int(''.join(list(filter(str.isdigit, str(x))))) for x in ax)
            label_list.append(ax)

    return label_list

# 画格子
def draw_label(img, label, block_size=cfg.TRAIN.RAW_BLOCK_SIZE):
    fig = plt.figure(figsize=[15,15])
    ax = plt.gca()
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    for l in label:
        rect = patches.Rectangle(((l[1]-1)*block_size, (l[0]-1)*block_size), block_size, block_size, linewidth=1,edgecolor='g',facecolor='none')  # 红色
        ax.add_patch(rect)
    plt.imshow(img)
    return len(label)
#     plt.savefig(save_dir+'/'+ os.path.split(image)[-1],dpi = 300,bbox_inches='tight',pad_inches = 0,transparent=False, frameon=False)