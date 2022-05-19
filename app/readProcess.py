#-*- coding: UTF-8 -*-
from multiprocessing import Process, Queue
import os
import time
import cv2
import numpy as np
import threading
import torch

class Process_queue():
    def __init__(self, maxsize =0):
        self.maxsize = maxsize
        self.q = Queue(maxsize)
        self._queue_End = False

    def setEnd(self):
        self.put(self._queue_End)

    def isEnd(self,item):
        if(item==self._queue_End):
            return True
        else:
            return False

    def put(self, item):
        try:
            self.q.put(item)
        except:
            print("Process_queue:put block!!!",os.getpid())
            pass

    def get(self):
        item = None
        try:
            item = self.q.get_nowait()
        except:
            pass
        return item

    def qsize(self):
        _len = self.q.qsize()
        return _len


class textReadProcess(Process):
    def __init__(self, read_queue,textfile):
        super().__init__()
        self.read_queue = read_queue
        self.textfile=textfile

    def parse_index(self, abs_path):
        # 是仁义设计的txt文件，不是fileindex
        # 仁义确定的这个文件，第一列是jpg文件名，第二列是图片全路径，第三列是标记全路径
        idx_path = abs_path
        try:
            if os.path.exists(idx_path) == False:
                print('exit')
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
                    #print(file_meta[0], img_file, label_file)
                    yield (file_meta[0], img_file, label_file)
            f.close()
        except IOError as err:
            pass ##print('File error:'+str(err))

    def run(self):
        print("readTEXT...",os.getpid())
        for item in self.parse_index(self.textfile):
            self.read_queue.put(item)
        self.read_queue.setEnd()
        print("read_queue setEnd!!!")


class imgReadProcess(Process):
    def __init__(self, read_queue,data_queue,thread_num=15):
        super().__init__()
        self.read_queue = read_queue
        self.data_queue = data_queue
        self.thread_num=thread_num
        self.cfg_ctx = {}
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
        #img /= 255.
        return img

    def data_read(self):
        count = 0
        while True:
            item = self.read_queue.get()
            if item is None:
                time.sleep(0.1)
                continue
            elif(item==False):
                break
            count += 1
            image = self.func_image_loader(item[1])
            if image is None:
                print("img is none")
                imgsize = (0, 0)
            else:
                imgsize = image.shape
            image = self.img_grid(image)

            label = self.func_label_loader(item[2])
            label = self.label_grid(label)
            ### 这个item就是仁义定义的那个结构，三个部分，jpg，全路径jpg，全路径txt，(多加了图像大小的参数)
            self.data_queue.put((image, label, item, imgsize))
            #print("put into data_queue...")
        time.sleep(5)
        self.data_queue.setEnd()
        print("data_queue setEnd")

    def run(self):
        #data_read函数完成数据读取工作，该部分增加多线程
        print("imgReadStart.........",os.getpid())
        thread_list=[]
        for i in range(self.thread_num):
            t=threading.Thread(target=self.data_read,)
            thread_list.append(t)
            t.setDaemon(True)
            t.start()
        self.data_read()
        for t in thread_list:
            t.join()
        print("imgread End!!!")

class dataSupplyProcess(Process):
    def __init__(self, data_queue,dataSupply_queue,batch_size=50):
        super().__init__()
        self.data_queue = data_queue
        self.dataSupply_queue = dataSupply_queue
        self.batch_size=batch_size
        self.cfg_ctx = {}
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
        #img /= 255.
        return img


    def run(self):
        print("dataSupplyStart.........",os.getpid())
        img_data = []
        label_data = []
        info_data=[]
        imgsize_data=[]
        count=0
        while True:
            item = self.data_queue.get()
            if item is None:
                time.sleep(0.1)
                continue
            elif(item==False):
                break
            # img_data.append(self.img_grid(item[0]))
            # label_data.append(self.label_grid(item[1]))
            img_data.append(item[0])
            label_data.append(item[1])
            #info_data.append(item[2])
            #imgsize_data.append(item[3])
            #print("##########")

            count=count+1
            if(count%self.batch_size==0):
                X = np.expand_dims(img_data, axis=1)
                X = torch.from_numpy(X)
                Y = np.expand_dims(label_data, axis=1)
                Y = torch.from_numpy(Y).float()
                #self.dataSupply_queue.put((X,Y,info_data,imgsize_data))
                self.dataSupply_queue.put((X, Y))
                #print("dataSupply_queue put into")
                print("data_queue_size:",data_queue.qsize())
                print("dataSupply_queue_size:", dataSupply_queue.qsize())
                img_data[:]= []
                label_data[:] = []
                #info_data[:] = []
                #imgsize_data[:] = []
                count = 0
        X = np.expand_dims(img_data, axis=1)
        X = torch.from_numpy(X)
        Y = np.expand_dims(label_data, axis=1)
        Y = torch.from_numpy(Y).float()
        self.dataSupply_queue.put((X, Y, info_data, imgsize_data))
        time.sleep(5)
        self.dataSupply_queue.setEnd()
        print("dataSupply_queue setEnd")




class trainProcess(Process):
    def __init__(self, dataSupply_queue,batch_size=50):
        super().__init__()
        self.dataSupply_queue = dataSupply_queue
        self.batch_size=batch_size

    def run(self):
        print("trainStart.........",os.getpid())
        count=0
        startTime=time.time()
        hundred_start=time.time()
        while True:
            item = self.dataSupply_queue.get()
            if item is None:
                time.sleep(0.1)
                continue
            elif(item==False):
                break
            #print(np.shape(item[0]))
            time.sleep(0.4)

            endTime=time.time()
            print("runTime:",endTime-startTime)
            startTime=endTime
            count=count+1
            if(count%100==0):
                print("###############hundred_runtime:",time.time()-hundred_start)
                hundred_start=time.time()
                time.sleep(5)
                count=0
        time.sleep(5)
        self.dataSupply_queue.setEnd()
        print("train End!!!!")






if __name__ == '__main__':
    read_queue = Process_queue()
    data_queue=Process_queue(1000)
    data_queue1 = Process_queue(1000)
    data_queue2 = Process_queue(1000)
    data_queue3 = Process_queue(1000)

    dataSupply_queue=Process_queue(100)
    #创建进程
    thread_num = 2
    readtxt_process=textReadProcess(read_queue,"./smalltrain.txt")
    imgread_process=imgReadProcess(read_queue,data_queue,thread_num=thread_num)
    imgread_process1 = imgReadProcess(read_queue, data_queue1, thread_num=thread_num)
    imgread_process2 = imgReadProcess(read_queue, data_queue2, thread_num=thread_num)
    imgread_process3 = imgReadProcess(read_queue, data_queue3, thread_num=thread_num)

    imgread_list=[]
    imgread_num=2

    for i in range(imgread_num):
        imgread_process = imgReadProcess(read_queue, data_queue)
        imgread_process.start()
        imgread_list.append(imgread_process)

    for i in range(imgread_num):
        imgread_process = imgReadProcess(read_queue, data_queue1)
        imgread_process.start()
        imgread_list.append(imgread_process)
    for i in range(imgread_num):
        imgread_process = imgReadProcess(read_queue, data_queue2)
        imgread_process.start()
        imgread_list.append(imgread_process)
    for i in range(imgread_num):
        imgread_process = imgReadProcess(read_queue, data_queue3)
        imgread_process.start()
        imgread_list.append(imgread_process)

    dataSupply_list=[]
    dataprocess_num=3
    for i in range(dataprocess_num):
        dataSupply_process=dataSupplyProcess(data_queue,dataSupply_queue)
        dataSupply_process.start()
        dataSupply_list.append(dataSupply_process)
    for i in range(dataprocess_num):
        dataSupply_process=dataSupplyProcess(data_queue1,dataSupply_queue)
        dataSupply_process.start()
        dataSupply_list.append(dataSupply_process)
    for i in range(dataprocess_num):
        dataSupply_process=dataSupplyProcess(data_queue2,dataSupply_queue)
        dataSupply_process.start()
        dataSupply_list.append(dataSupply_process)
    for i in range(dataprocess_num):
        dataSupply_process=dataSupplyProcess(data_queue3,dataSupply_queue)
        dataSupply_process.start()
        dataSupply_list.append(dataSupply_process)


    train_process=trainProcess(dataSupply_queue)

    #进程开始

    readtxt_process.start()
    train_process.start()

    #进程join
    readtxt_process.join()
    for imgread in imgread_list:
        imgread.join()
    for dataSupply in dataSupply_list:
        dataSupply.join()

    train_process.join()
