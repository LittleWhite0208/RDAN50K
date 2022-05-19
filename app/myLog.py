import logging
import sys

import os

class myLog():
    def __init__(self):
        self.logger=self.log_set()

    def log_set(self):

        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)  # 设置Level
        handler = logging.FileHandler("log.txt") #"log.txt"
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 输出的格式
        handler.setFormatter(formatter)
        # 控制台的设置
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        # 绑定文件和控制台
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger

    def infolog(self,msg):
        self.logger.info(msg)
    def errorlog(self,msg):
        self.logger.error(msg)

mylog=myLog()
def log_info(msg):
    mylog.infolog(msg)
def log_error(msg):
    mylog.errorlog(msg)


if __name__=="__main__":
    mylog=myLog()
    mylog.infolog("hello")
    mylog.errorlog("world!")
