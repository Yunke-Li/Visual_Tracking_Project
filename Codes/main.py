import numpy as np
import os
import math
from utils import GetImgNameByEveryDir

# set path
path = '..\\data'
# FileName, FileNameWithPath, FileDir = GetImgNameByEveryDir(path)
# initialization
inistate = [161, 65, 75, 95]
pos = [inistate[1]+inistate[3]/2, inistate[0]+inistate[2]/2]
target_size = [inistate[3], inistate[2]]
# parameters according to the paper
padding = 1
rho = 0.075
size_x = math.floor(target_size[0]*(1+padding))
size_y = math.floor(target_size[1]*(1+padding))
# parameters of scale update. See Eq.(15)
scale = 1
lam = 0.25
num = 5
# store pre-computed confidence map
alpha = 2.25
x = np.arange(1, size_x+1, 1) - np.floor(size_x/2)
y = np.arange(1, size_y+1, 1) - np.floor(size_y/2)
rs, cs = np.meshgrid(x, y)
rs_t = np.transpose(rs)
cs_t = np.transpose(cs)
dist = rs_t * rs_t + cs_t * cs_t
conf = np.exp(-0.5 / alpha * np.sqrt(dist))
conf = conf/np.sum(conf)
conf_fft = np.fft.fft2(conf)
# store pre-computed weight window
