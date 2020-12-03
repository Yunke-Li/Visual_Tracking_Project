import math
import numpy as np


def get_context(im, pos, sz, window):
    x = [pos[1] + i - (sz[1]/2) for i in range(1, sz[1]+1)]
    y = [pos[0] + j - (sz[0]/2) for j in range(1, sz[0]+1)]
    xs = [math.floor(i) for i in x]
    ys = [math.floor(j) for j in y]

    for i in xs:
        if i < 1:
            i = 1

    for j in ys:
        if j < 1:
            j = 1

    for i in xs:
        if i > im.shape(1):
            i = im.shape(1)

    for j in ys:
        if j > im.shape(0):
            j = im.shape(0)

    out = im[ys, xs, :]
    out = out - np.mean(out)
    out = window * out

    return out
