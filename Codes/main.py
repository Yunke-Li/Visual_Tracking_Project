import numpy as np
import os
import math
import cv2
from utils import GetImgNameByEveryDir, get_context

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
hamming_window = np.hamming(size_x).reshape(size_x, 1) * np.hanning(size_y).reshape(size_y, 1).transpose()
hamming_window = hamming_window.astype('double')
sigma = (target_size[0] + target_size[1])/2
window = hamming_window * np.exp(-0.5/(sigma*sigma)*dist)
window = hamming_window/hamming_window.sum()
#
files = os.listdir(path)   # 读入文件夹
num_img = len(files)
for frame in range(1, num_img):
    sigma = sigma * scale
    window = hamming_window * np.exp(-0.5 / (sigma * sigma) * dist)
    window = hamming_window / hamming_window.sum()
    # load image
    img = cv2.imread(path + '\\' + files[frame-1])
    if img[2] > 1:
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contextprior = get_context(im, pos, sz, window) #### sz?
    #
    if frame > 1:
        # calculate response of the confidence map at all locations
        confmap = (np.fft.ifft2(Hstcf * np.fft.fft2(contextprior))).real()
        #target location is at the maximum response
        [row, col] = np.unravel_index(np.argmax(confmap), confmap.shape)
        row = row + 1
        col = col + 1
        pos = pos - [size_x/2, size_y/2] + [row, col]
        contextprior = get_context(im, pos, sz, window) ## sz?
        maxconf[frame-2] = np.max(conftmp)
        # update scale by Eq.(15)
        if frame % (num + 2) == 0:
            scale_curr = 0
            for kk in range(1, num):
                scale_curr = scale_curr + np.sqrt(maxconf[frame-kk-1]/maxconf[frame-kk-2])
            scale = (1 - lam) * scale + lam * (scale_curr/num)

    # update the spatial context model h^sc in Eq.(9)
    contextprior = get_context(im, pos, sz, window) # sz??
    hscf = conf_fft / np.fft.fft2(contextprior + np.finfo(float).eps)
    # update the spatio-temporal context model by Eq.(12)
    if frame == 1:
        Hstcf = hscf
    else:
        # update the spatio-temporal context model H^stc by Eq.(12)
        Hstcf = (1 - rho) * Hstcf + rho * hscf

    # visualization
    target_size










print()
