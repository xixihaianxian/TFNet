import random
import os
import numpy as np
import shutil
from tqdm import tqdm
import imageio
import cv2
import csv
import DataProcessMoudle
from typing import Dict

# 设置随机种子
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为python环境设置随机种子
    np.random.seed(seed)

# 数据预处理
def main(dataPath:str, saveDataPath:str)->None:
    r"""
    dataPath:原来数据的位置
    saveDataPath:保存处理之后数据的位置
    """
    fileTypes = sorted(os.listdir(dataPath)) #fileTypes=["test","dev","train"]
    framesList = []
    fpsList = []
    videoTimeList = []
    resolutionList = []
    for fileType in fileTypes:
        filePath = os.path.join(dataPath, fileType)
        saveFilePath = os.path.join(saveDataPath, fileType)
        translators = sorted(os.listdir(filePath))

        for translator in translators:
            translatorPath = os.path.join(filePath, translator)
            saveTranslatorPath = os.path.join(saveFilePath, translator)
            videos = sorted(os.listdir(translatorPath))

            for video in tqdm(videos):
                videoPath = os.path.join(translatorPath, video)
                nameString = video.split(".")
                saveImagePath = os.path.join(saveTranslatorPath, nameString[0])

                if not os.path.exists(saveImagePath):
                    os.makedirs(saveImagePath)
                # 获取视频的信息
                vid = imageio.get_reader(videoPath)  # 读取视频
                # nframes = vid.get_meta_data()['nframes']
                nframes = vid.count_frames() # 获取总总帧数
                fps = vid.get_meta_data()['fps'] # 获取fps(每秒里面含有多少帧)
                videoTime = vid.get_meta_data()['duration']# 视频的时长(单位s)
                resolution = vid.get_meta_data()['size']# 视频的分辨率(width,height)

                # 保存相关信息
                framesList.append(nframes)
                fpsList.append(fps)
                videoTimeList.append(videoTime)
                resolutionList.append(resolution)

                for i in range(nframes): # 循环视频的所有帧
                    try:
                        image = vid.get_data(i) # 取出视频的第 i 帧，得到一个图像数组。
                        # 将BGR模式转化为RBG模式
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # 改变形状大小
                        image = cv2.resize(image, (256, 256))

                        nameString = str(i)
                        for i in range(5 - len(nameString)):
                            nameString = "0" + nameString

                        imagePath = os.path.join(saveImagePath, nameString + ".jpg")
                        cv2.imencode('.jpg', image)[1].tofile(imagePath)
                    except:
                        print(nframes)
                        print(videoPath)

                vid.close() # 关闭视频文件并释放资源

    maxframeNum = max(framesList)
    minframeNum = min(framesList)
    maxVideoTime = max(videoTimeList)
    minVideoTime = min(videoTimeList)
    fpsSet = set(fpsList)
    resolutionSet = set(resolutionList)

    print(f"Max Frames Number:{maxframeNum}\n"
          f"Min Frames Number:{minframeNum}\n"
          f"Max Video Time:{maxVideoTime}\n"
          f"Min Video Time:{minVideoTime}\n"
          f"Fps Set:{fpsSet}\n"
          f"Resolution Set:{resolutionSet}\n")


if __name__ == '__main__':
    # dataPath = "/home/lj/lj/program/python/DataSets/CE-CSL/video"
    dataPath = "/usr/Sign-Language-Recognition/CE-CSL/video"
    # saveDataPath = "/home/lj/lj/program/python/DataSets/CE-CSL/video2"
    saveDataPath = "/usr/Sign-Language-Recognition/CE-CSL/video2"

    seed_torch()
    main(dataPath, saveDataPath)