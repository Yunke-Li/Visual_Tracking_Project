import numpy as np
import os
import math
import cv2
from utils import get_context

# set path
path = '..\\data'
# initialization
inistate = [161, 65, 75, 95]
pos = [inistate[1]+inistate[3]/2, inistate[0]+inistate[2]/2]
target_size = [inistate[3], inistate[2]]
target_size = np.array(target_size)
# parameters according to the paper
padding = 1
rho = 0.5
size_x = math.floor(target_size[0]*(1+padding))
size_y = math.floor(target_size[1]*(1+padding))
# parameters of scale update. See Eq.(15)
scale = 1
lam = 0.25
num = 30
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
window = hamming_window/window.sum()
#
files = os.listdir(path)   # 读入文件夹
num_img = len(files)
maxconf = np.empty([num_img-1])
conftmp = []

for frame in range(1, num_img+1):
    sigma = sigma * scale
    window = hamming_window * np.exp(-0.5 / (sigma * sigma) * dist)
    window = hamming_window / window.sum()
    # load image
    img = cv2.imread(path + '\\' + files[frame-1])
    if img.shape[2] > 1:
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sz = [size_x, size_y]
    contextprior = get_context(im, pos, sz, window)
    #
    if frame > 1:
        # calculate response of the confidence map at all locations
        confmap = np.fft.ifft2(Hstcf * np.fft.fft2(contextprior))
        confmap = np.real(confmap)
        #target location is at the maximum response
        [row, col] = np.unravel_index(np.argmax(confmap), confmap.shape)
        row = row + 1
        col = col + 1
        pos = [pos[0]-size_x/2+row, pos[1]-size_y/2+col]
        sz = [size_x, size_y]
        print('frame:', frame)
        # print('pos:', pos)
        # print('row:', row)
        # print('col:', col)
        contextprior = get_context(im, pos, sz, window)
        conftmp = np.fft.ifft2(Hstcf * np.fft.fft2(contextprior))
        conftmp = np.real(conftmp)
        maxconf[frame-2] = np.max(conftmp)

        # update scale by Eq.(15)
        if frame % (num + 2) == 0:
            scale_curr = 0
            for kk in range(1, num+1):
                scale_curr = scale_curr + np.sqrt(maxconf[frame-kk-1]/maxconf[frame-kk-2])
            scale = (1 - lam) * scale + lam * (scale_curr/num)

    # update the spatial context model h^sc in Eq.(9)
    sz = [size_x, size_y]
    contextprior = get_context(im, pos, sz, window)
    hscf = conf_fft / np.fft.fft2(contextprior + np.finfo(float).eps)
    # update the spatio-temporal context model by Eq.(12)
    if frame == 1:
        Hstcf = hscf
    else:
        # update the spatio-temporal context model H^stc by Eq.(12)
        Hstcf = (1 - rho) * Hstcf + rho * hscf

    # visualization
    target_size = target_size * scale
    pos_arr = np.array(pos)
    target_size_arr = np.array(target_size)
    rect_ori = np.flip((pos_arr - target_size_arr/2)) # origin of rectangle
    w = target_size[1]
    h = target_size[0]
    p1_x = int(rect_ori[0])
    p1_y = int(rect_ori[1])
    p2_x = int(p1_x + w)
    p2_y = int(p1_y + h)
    w = target_size[1]
    h = target_size[0]
    result = cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (255, 0, 0), 2)
    result_resize = cv2.resize(result, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)
    cv2.imshow('result', result_resize)
    cv2.waitKey(40)

