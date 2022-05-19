#-*- coding: UTF-8 -*- 
from __future__ import print_function
import os
import time
import torch.multiprocessing as multiprocessing


class _queue():
    def __init__(self, maxsize = 0):
        self.maxsize = maxsize
        self._queue_end = False
        self.q = multiprocessing.Queue(maxsize)

    def setEnd(self, _queue_end=True):
        self._queue_end = _queue_end

    def isEnd(self,item):
        if(item==self._queue_end):
            return True
        else:
            return False

    def put(self, item):
        try:
            self.q.put(item,timeout=50)
        except:
            print("_queue:put block!!!",os.getpid())
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
