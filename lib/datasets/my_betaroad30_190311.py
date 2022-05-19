'''
利用dataloader完成图片的读取
'''
import sys
sys.path.append("../../")


from torch.utils import data
import os
import numpy as np
import yaml
import datasets.constant_cfg as ccfg
import cv2
from numpy.lib import pad
from torch.utils.data.dataloader import default_collate




DEBUGPRINT=False

def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    try:
        batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    except:
        pass
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据

class BetaroadDataset30(data.Dataset):
    def __init__(self,filename,DISEASE_TYPE):
        self._filename=filename
        self.data=self._load_from_file(self._filename)
        self.disease_type=DISEASE_TYPE

    def __getitem__(self,index):
        #step1：获得图片，标记的全路径文件名
        imgpath=self.data[index][1]
        labelpath=self.data[index][2]
        if DEBUGPRINT==True:
            print("img:",imgpath)
            print("labe:",labelpath)

        #step2:对图片进行处理，得到能直接放入dataloader的数据
        img=self.imgProcessing(imgpath)
        #step3:对标记进行处理，得到能直接放入dataloader的数据
        label=self.labelProcessing(labelpath)
        if DEBUGPRINT==True:
            print("imgsize:",np.shape(img))
            print("labelsize:",np.shape(label))

        if (img is None):
            #print("img is None...index=%d"%index)
            #self.logger.errorlog("img is None,index=%d" %index)
            #log_error("img is None,index=%d" %index)
            return None,None
        if(label is None):
            #print("label is None...")
            #self.logger.errorlog("img is None,index=%d" % index)
            #log_error("img is None,index=%d" % index)
            return None, None

        return img,label

    def __len__(self):
        return len(self.data)

    def _load_from_file(self,filename):
        #从file文件中将内容读取出来，
        # 并进行分析处理，返回一个三元组[name,imgpath,labelpath]的list列表
        idx_path = filename
        content=[]
        try:
            if os.path.exists(idx_path) == False:
                #self.logger.errorlog("file doesn't exist,filename=%s" % filename)
                #log_error("file doesn't exist,filename=%s" % filename)
                return
            count = 0
            f = open(idx_path, "r")
            for line in f:
                count += 1
                fileTmp = line.replace('\r', '').replace('\n', '')
                file_meta = fileTmp.split(",")
                bound = len(file_meta)
                # print('from cry file:%d'%count)
                if bound >= 3:
                    img_file = os.path.join(file_meta[1], file_meta[0])
                    label_file = os.path.join(file_meta[2], file_meta[0])
                    label_file = label_file.replace('.jpg', '.txt')
                    # 这个需要注意：分量是，文件名，全路径图片名，全路径标记名
                    # print(file_meta[0], img_file, label_file)
                    #yield (file_meta[0], img_file, label_file)
                    content.append((file_meta[0], img_file, label_file))
            f.close()
            return content
        except IOError as err:
            #self.logger.errorlog("load_from_file failed,filename=%s" % filename)
            #log_error("load_from_file failed,filename=%s" % filename)
            pass  ##print('File error:'+str(err))


    def imgProcessing(self,imgpath):
        #对图片进行处理，能够得到图片矩阵信息直接放到dataloader
        #包括图片读取，图片填充压缩等等
        try:
            #step1:读取图片
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            #step2：图片处理
            img=self.img_grid(img)
            #step3:升维
            img=np.expand_dims(img,axis=0)
            return img
        except:
            #self.logger.errorlog("imgProcessing failed,filename=%s" % imgpath)
            #log_error("imgProcessing failed,filename=%s" % imgpath)
            return None

    def img_grid(self, img):
        if (img is None):
            #self.logger.errorlog("img_grid  failed,img is None" )
            #log_error("img_grid  failed,img is None")
            return None

        if ccfg.RAW_IMAGE_ROWS < img.shape[0]:
            img = img[:ccfg.RAW_IMAGE_ROWS, :]

        if ccfg.RAW_IMAGE_COLS< img.shape[1]:
            img = img[:, :ccfg.RAW_IMAGE_COLS]

        img = pad(img, (
        (0, ccfg.RAW_IMAGE_ROWS - img.shape[0]), (0, ccfg.RAW_IMAGE_COLS - img.shape[1])),
                  'constant', constant_values=255)
        img = cv2.resize(img, (ccfg.SHRINK_IMG_W, ccfg.SHRINK_IMG_H),
                         interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')  # 转换类型为float32
        img -= np.mean(img)
        img /= 255.
        return img


    def labelProcessing(self,labelpath):
        #对标签进行处理，能够直接得到标记的矩阵信息房贷dataloader中
        #包括标记的识别读取，转化成对应的矩阵信息
        try:
            #step1:读取label文件，将内容保存到lines内
            with open(labelpath, "r", encoding='utf-8') as l:
                lines = l.readlines()
            # 一个文件的标记就是一个数组
            #step2:对lines进行解析，将病害类型的二元组提取出来
            label_list = self.extract_label(lines,self.disease_type)
            #step3:对label_list进行解析，得到label的二维矩阵信息
            label=self.label_grid(label_list)
            #step4:升维
            label=np.expand_dims(label,axis=0)
            return label
        except:
            #print("labelProcessing return None==read path is None...")
            #self.logger.errorlog("labelProcessing failed,filename=%s" % labelpath)
            #log_error("labelProcessing failed,filename=%s" % labelpath)
            return None

    def extract_label(self, lines,DISEASE_TYPE):
        if(DISEASE_TYPE=="CRACK"):
            label_type=ccfg.LABEL_CRACK
            label_type2 = ccfg.LABEL_CRACK2
        elif(DISEASE_TYPE=="REPAIR"):
            label_type=ccfg.LABEL_REPAIR
            label_type2 = ccfg.LABEL_REPAIR2
        else:
            #print("extrac_label return None...")
            #self.logger.errorlog("extrac_label return None,DISEASE_TYPE=%s" % DISEASE_TYPE)
            #log_error("extrac_label return None,DISEASE_TYPE=%s" % DISEASE_TYPE)
            return None
        if DEBUGPRINT==True:
            print("DISEASE_TYPE:",DISEASE_TYPE)
            print("label_type:",label_type)
        ret = []
        for line in lines:
            if line.find(label_type) != -1:
                ax = line.strip("\n").replace(label_type2, "").split(",")
                ax = tuple(int(x) for x in ax)
                ret.append(ax)
        return ret

    def label_grid(self,label):
        if label==None:
            #print("label_grid return None...")
            #self.logger.errorlog("label_grid failed,label is None" )
            #log_error("label_grid failed,label is None")
            return None
        grid = np.zeros((ccfg.IMG_BLOCK_H, ccfg.IMG_BLOCK_W), np.uint8)
        # 将含有坐标的列表，转换成二维数组个，这个三位的，但是实际上是两个维度的
        if label is not None:
            for l in label:
                if l[0] <= ccfg.IMG_BLOCK_H and l[1] <= ccfg.IMG_BLOCK_W:
                    grid[l[0] - 1, l[1] - 1] = 1
        return grid



if __name__=="__main__":
    dataset=BetaroadDataset30("/home/cry/datafileindex/9000whitetest.txt","CRACK")
    dataset.__getitem__(1)
    print("len:",dataset.__len__())


