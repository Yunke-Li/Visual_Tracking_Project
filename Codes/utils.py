import numpy
import os


def GetImgNameByEveryDir(file_dir):
    FileNameWithPath = []
    FileName = []
    FileDir = []
    videoProperty = ['.jpg']
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in videoProperty:
                FileNameWithPath.append(os.path.join(root, file))  # path
                FileName.append(file)  # name
                FileDir.append(root[len(file_dir):])  # folder
                # FileNameWithPath.sort(key=lambda x:int(x.split('.')[0]))
                FileName.sort(key=lambda x:int(x.split('.')[0]))
                # FileDir.sort(key=lambda x:int(x.split('.')[0]))
    return FileName, FileNameWithPath, FileDir
