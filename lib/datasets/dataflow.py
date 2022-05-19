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
#from mylog import myPrint
'''''
该部分主要进行数据的读取,对输入的train.txt，test.txt两个filename进行操作
在train.py中调用，每次得到指定数量的item（img，label，info）
'''
class _dict_mem():
    def __init__(self):
        self._dict_in_mem = {}
        self._dict_in_mem_size = 1000
        self.threadLock = threading.Lock()

    def full(self):
        bfull = False
        self.threadLock.acquire()
        _dict_len = len(self._dict_in_mem)
        if _dict_len > self._dict_in_mem_size:
            bfull = True
        self.threadLock.release()
        return bfull

    def put(self, key, item):
        self.threadLock.acquire()
        self._dict_in_mem[key] = item
        self.threadLock.release()

    def get(self, key):
        item = None
        self.threadLock.acquire()
        try:
            item = self._dict_in_mem[key]
            del self._dict_in_mem[key]
        except Exception as err:
            #myPrint(err)
            pass
        self.threadLock.release()
        return item

    def exist(self, key):
        bexist = False
        self.threadLock.acquire()
        if key in self._dict_in_mem:
            bexist = True
        self.threadLock.release()
        return bexist

    def size(self):
        _dict_len = 0
        self.threadLock.acquire()
        _dict_len = len(self._dict_in_mem)
        self.threadLock.release()
        return _dict_len


## get阻塞
class _queue():
    def __init__(self, maxsize =0):
        self._queue_End = False
        self.maxsize = maxsize
        self.q = queue.Queue(maxsize)
        self.threadLock = threading.Lock()

    def setEnd(self):
        self.threadLock.acquire()
        self._queue_End = True
        self.threadLock.release()

    def isEnd(self):
        return self._queue_End

    def full(self):
        _full = True
        self.threadLock.acquire()
        if self.maxsize <= 0 or self.q.qsize()< self.maxsize:
            _full = False
        self.threadLock.release()
        return _full

    def put(self, item):
        self.threadLock.acquire()
        self.q.put(item)
        self.threadLock.release()

    def clear(self):
        self.threadLock.acquire()
        while self.q.empty() is False:
            self.q.get()
        self.threadLock.release()

    def get(self):
        item = None
        self.threadLock.acquire()
        try:
            item = self.q.get(False)
        except:
            pass
        self.threadLock.release()
        return item

    def qsize(self):
        _len = 0
        self.threadLock.acquire()
        _len = self.q.qsize()
        self.threadLock.release()
        return _len

## 这里filename_lists里面是tuple，三个元素，一个是txt，一个是
## 图的目录，一个是label的目录
class parseThread(threading.Thread):
    def __init__(self, threadID, name, _data_loader):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self._data_loader = _data_loader
        self.stopFlag = False

    def stop_parse_thread(self):
        self.setStopFlag(True)

    def setStopFlag(self, _stopFlag=True):
        self.stopFlag = _stopFlag
        
    def waitThread(self):
        while True:
            if self.is_alive():
                time.sleep(0.1)
            else:
                break

    def run(self):
        for item in self.parse_index(self._data_loader.datafile):
            while self._data_loader.read_queue.full():
                time.sleep(0.1)
            self._data_loader.read_queue.put(item)
            self._data_loader.image_queue.put(item[1])
        self._data_loader.read_queue.setEnd()
        self._data_loader.image_queue.setEnd()

    # 这个函数得到的是图片文件名和全路径的文件名对照
    def parse_index(self, abs_path):
        # 是仁义设计的txt文件，不是fileindex
        # 仁义确定的这个文件，第一列是jpg文件名，第二列是图片全路径，第三列是标记全路径
        idx_path = abs_path

        try:
            if os.path.exists(idx_path) == False:
                ##print('exit')
                return

            count = 0
            f = open(idx_path, "r")
            for line in f:
                count += 1
                fileTmp = line.replace('\r','').replace('\n','')
                file_meta = fileTmp.split(",")
                bound = len(file_meta)
                #print('from cry file:%d'%count)
                if bound >= 3:
                    img_file = os.path.join(file_meta[1], file_meta[0])
                    label_file = os.path.join(file_meta[2], file_meta[0])
                    label_file = label_file.replace('.jpg','.txt')
                    # 这个需要注意：分量是，文件名，全路径图片名，全路径标记名
                    yield (file_meta[0], img_file, label_file)
            f.close()
        except IOError as err:
            pass ##print('File error:'+str(err))



class readThread(threading.Thread):
    def __init__(self, threadID, name, _data_loader):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self._data_loader = _data_loader
        self.stopFlag = False
        self.cfg_ctx={}
        self.cfg_ctx['RAW_IMAGE_ROWS'] = 2200
        self.cfg_ctx['RAW_IMAGE_COLS'] = 3400
        # 原始块大小，长宽皆为100像素
        self.cfg_ctx['RAW_BLOCK_SIZE'] = 100
        # 压缩后块大小为长宽32像素
        self.cfg_ctx['SHRINK_BLOCK_SIZE'] = 32
        self.cfg_ctx['RAW_IMG_HEIGHT'] = self.cfg_ctx['RAW_IMAGE_ROWS']
        self.cfg_ctx['RAW_IMG_WIDTH'] = self.cfg_ctx['RAW_IMAGE_COLS']
        # 压缩后图像像素为672*1024
        # 修改为704*1088
        self.cfg_ctx['SHRINK_IMG_HEIGHT'] = 704
        self.cfg_ctx['SHRINK_IMG_WIDTH'] = 1088
        # 块的高和宽，总共21*32块
        # 修改为22*34
        self.cfg_ctx['IMG_BLOCK_H'] = 22
        self.cfg_ctx['IMG_BLOCK_W'] = 34

    def setStopFlag(self, _stopFlag):
        self.stopFlag = _stopFlag
        
    def waitThread(self):
        while True:
            if self.is_alive():
                time.sleep(0.1)
            else:
                break

    def run(self):
        start = time.time()
        count = 0
        while True:
            if self.stopFlag is True:
                break

            if self._data_loader.img_dict_mem.full():
                #print('img_dict_mem.is full')
                time.sleep(0.1)
                continue

            item = self._data_loader.read_queue.get()
            #print('%d,%d,%s'%(self._data_loader.read_queue.qsize(),self._data_loader.img_dict_mem.size(),self.name))
            
            if item is None:
                if self._data_loader.read_queue.isEnd():
                    break                  
                else:
                    time.sleep(0.1)
                    continue
               
            count += 1
            loadtime=time.time()
            image = self.func_image_loader(item[1])
            loadend=time.time()
            #print("%f,%s"%((loadend-loadtime),self.name))
            if image is None:
                print("img is none")
                imgsize=(0,0)
            else:
                imgsize = image.shape
            image=self.img_grid(image)

            label = None
            if self._data_loader.onlyTrain is True:
                label = self.func_label_loader(item[2])
            label=self.label_grid(label)
            ### 这个item就是仁义定义的那个结构，三个部分，jpg，全路径jpg，全路径txt，(多加了图像大小的参数)
            self._data_loader.img_dict_mem.put(item[1], (image, label, item,imgsize))


    # 读取出来的图是2维的图，不是三维，此处不进行扩充
    def func_image_loader(self, path):
        #print(path)
        try:
            if not os.path.exists(path):
                return None
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return img
        except:
            return None

    # 这里直接就是npy的数据，得到就是二维的数组
    def func_label_loader(self, fileTmp):
        #print(path)
        try:
            if not os.path.exists(fileTmp):
                return None

            if '.txt' in fileTmp:
                return self._label_loader(fileTmp)

            label_array = np.load(fileTmp)
            return label_array
        except:
            return None

    def _label_loader(self, lbl_filepath):
        label_list = []
        # 载入人工标记的文本，k是文件名，v是全路径的jpg文件名
        # 读入一个人工标记的文件？标记可能很多，如果需要扩展成多种标签，那么扩展ret就可以
        try:
            if not os.path.exists(lbl_filepath):
                return None

            with open(lbl_filepath, "r", encoding='utf-8') as l:
                lines = l.readlines()
            # 一个文件的标记就是一个数组
            label_list = self.extract_label(lines)
        except Exception as err:
            return None
        return label_list

    def extract_label(self, lines):
        return self._extract_label_crack(lines)

    # 从标记文件中获取裂缝的坐标信息，解析成一个列表，这个
    # 列表的每个元素是一个tuple，每个tuple就是x,y坐标
    def _extract_label_crack(self, lines):
        ret = []
        for line in lines:
            if line.find("Bad_BlockPos") != -1:
                ax = line.strip("\n").replace('Bad_BlockPos:', "").split(",")
                ax = tuple(int(x) for x in ax)
                ret.append(ax)
        return ret

    # 从标记文件中获取修补的坐标信息，解析成一个列表，这个
    # 列表的每个元素是一个tuple，每个tuple就是x,y坐标
    def _extract_label_repair(self, lines):
        ret = []
        for line in lines:
            if line.find("Bad_RepairBlockPos") != -1:
                ax = line.strip("\n").replace("Bad_RepairBlockPos:", "").split(",")
                ax = tuple(int(x) for x in ax)
                ret.append(ax)
        return ret


    def label_grid(self,label):
        grid = np.zeros((self.cfg_ctx['IMG_BLOCK_H'], self.cfg_ctx['IMG_BLOCK_W']), np.uint8)
        # 将含有坐标的列表，转换成二维数组个，这个三位的，但是实际上是两个维度的
        if label is not None:
            for l in label:
                if l[0] <= self.cfg_ctx['IMG_BLOCK_H'] and l[1] <= self.cfg_ctx['IMG_BLOCK_W']:
                    grid[l[0] - 1, l[1] - 1] = 1
        return grid
    def img_grid(self,img):
        if(img is None):
            img = np.zeros((self.cfg_ctx['RAW_IMAGE_ROWS'],self.cfg_ctx['RAW_IMAGE_COLS']))
            img[:]=255

        if self.cfg_ctx['RAW_IMAGE_ROWS'] < img.shape[0]:
            img = img[:self.cfg_ctx['RAW_IMAGE_ROWS'], :]

        if self.cfg_ctx['RAW_IMAGE_COLS'] < img.shape[1]:
            img = img[:,:self.cfg_ctx['RAW_IMAGE_COLS']]
        if (img.shape[0] > self.cfg_ctx['RAW_IMAGE_ROWS'] or img.shape[1] > self.cfg_ctx['RAW_IMAGE_COLS']):
            img = pad(img, ((0, self.cfg_ctx['RAW_IMAGE_ROWS'] - img.shape[0]),(0, self.cfg_ctx['RAW_IMAGE_COLS'] - img.shape[1])), 'constant',constant_values=255)
        img = cv2.resize(img, (self.cfg_ctx['SHRINK_IMG_WIDTH'], self.cfg_ctx['SHRINK_IMG_HEIGHT']), interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')  # 转换类型为float32
        img -= np.mean(img)
        img /= 255.
        return img


# 这个类就是动态的读取fileidx_lists的图像和mask图
# 可以是多个目录（绝对路径），每一个目录下面必须有一个fileindex
# 读出原始的图片和mask图片
class data_loader():
    def __init__(self, datafile, sType='train', isOnlyTrain=True):
		# 中间项目可能为tuple，包含原图和mask图的绝对路径，本次没有这个问题
        self.datafile = datafile
        self.sType = sType
        ## 存的是三元组，jpg文件名、全路径jpg，全路径标记
        self.read_queue = _queue()
        ## 只有jpg文件名，作为关键字key
        self.image_queue = _queue()

        self.img_dict_mem = _dict_mem()
        self.line_size = self.getLines()
        self.onlyTrain = isOnlyTrain

        self.read_thread_lists = []
        self.parse_thread = None
        



    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
    
    def get_XY(self, image):
        item = [0,0]
        if image is not None:
            item[0] = image.shape[0] ## height
            item[1] = image.shape[1] ## width
        return item

    def getLines(self):
        count = 0
        try:
            f = open(self.datafile, "r")
            for line in f:
                count += 1
            f.close()
        except IOError as err:
            print('getLines:File error:'+str(err))
        #print('file line number:%d'%count)
        return count

    def get_size(self):
        return self.line_size

    def wait_read_thread(self):
        for i in range(len(self.read_thread_lists)):
            self.read_thread_lists[i].waitThread()

    def stop_read_thread(self):
        for i in range(len(self.read_thread_lists)):
            #self._async_raise(self.read_thread_lists[i].threadID, SystemExit)
            self.read_thread_lists[i].setStopFlag(True)

    def get_thread_num(self):
        if self.line_size >= 50000:
            return 11
        elif self.line_size >= 10000:
            return 5
        elif self.line_size >= 5000:
            return 3
        else:
            return 1



        


    # 启动读取线程时，需要把读取的queue充满，否则无法确定启动
    # 多少个线程
    def launch_read_thread(self):
        a = random.randint(1, 254)
        #threads_sum = self.get_thread_num()
        for num in range(self.get_thread_num()):
            rd_thread = readThread(a+num, self.sType+'readThread'+str(num), self)
            rd_thread.setDaemon(True)
            rd_thread.start()
            self.read_thread_lists.append(rd_thread)

    # 一次只yield一张图片，这是最为重要的接口
    #不同的框架，利用这个yield，可以采用自己需要的方式
    #我这里只是一次给一张图/标记
    ## 再次调用？
    def flow_one(self):
        self.read_queue.clear()
        self.image_queue.clear()
        self.parse_thread = parseThread(101, self.sType+'parseThread1', self)
        self.parse_thread.setDaemon(True)
        self.parse_thread.start()

        self.launch_read_thread()

        while True:
            key = self.image_queue.get()
            if key is None:
                if self.image_queue.isEnd():
                    break
                else:
                    time.sleep(0.1)
                    continue
            #print(self.img_dict_mem.size())
            while True:
                if self.img_dict_mem.exist(key):
                    break
                #print('key=%s is not exist!!'%key)
                time.sleep(0.1)
            
            item = self.img_dict_mem.get(key)
            # 给出去的是一个tuple，三个部分：raw图，一个标记（如果识别这个是None），一个tuple（三元组，jpg，全路径jpg，标记全路径）
            yield item

        self.stop_read_thread()
        self.parse_thread.stop_parse_thread()
        #self._async_raise(self.parse_thread.threadID, SystemExit)
        


    ##### 用来处理画图的路径
    # 默认以10张图片为一批次
    ## 这里给出的是两个原始的
    def flow(self, batch_size=10):
        num = 0
        count = 0
        data_batch = []
        for item in self.flow_one():
            count += 1
            num += 1
            data_batch.append(item)
            if count >= batch_size:
                yield data_batch
                data_batch[:] = []
                count = 0
        if count > 0: # 处理最后一个batch，大小不及batch_size
            yield data_batch
        #print('Finally size = %d'%num)
            
            


# 本次修改，主要集中在产生的日志文件没有两个了，只有一个log结尾的
# 如果用bash启动，那么就产生一个和result目下一样的日志文件
# 如果直接python启动，那么就没有日志文件，result下的目录之自动产生
# 就不能和日志一致了
# 推荐用bash train.sh启动
if __name__ == '__main__':
    datafile = '/home/xgs/datafileindex/whitecracktrain.txt'
    _loader = data_loader(datafile)

    count = 0
    countNoLabel = 0
    batch_size = 0
    sumsize = _loader.get_size()
    starttime = time.time()
    while True:
        count = 0
        for items in _loader.flow(50):
            endtime = time.time()
            #time.sleep(0.3)
            print("runtime:%.3f  %d" % (endtime - starttime, count))
            count = count + 1
            starttime = endtime


            #
        # for item in items:
        #     count += 1
        #     raw_image_hw = _loader.get_XY(item[0])
        #     label_list = []
        #     if item[1] is not None:
        #         for it in item[1]:
        #             axis = '(%d,%d)'%(it[0],it[1])
        #             label_list.append(axis)
        #         labels = ','.join(label_list)
        #     else:
        #         countNoLabel += 1
        #     print('image fileName=%s, raw_image_size = (%d,%d), label =[%s], (%d:%d:%d:%d).'
        #         % (item[2][0], raw_image_hw[0], raw_image_hw[1], labels, countNoLabel, count, sumsize,batch_size))

