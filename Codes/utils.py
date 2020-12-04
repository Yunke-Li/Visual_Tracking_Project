import numpy as np


def get_context(im, pos, sz, window):
    x = [pos[1] + i - (sz[1] / 2) for i in range(1, sz[1] + 1)]
    y = [pos[0] + j - (sz[0] / 2) for j in range(1, sz[0] + 1)]
    xs = np.floor(x)
    ys = np.floor(y)

    # for i in xs:
    #     if i < 1:
    #         i = 1
    xs[xs < 1] = 1

    # for j in ys:
    #     if j < 1:
    #         j = 1
    ys[ys < 1] = 1

    # for i in xs:
    #     if i > im.shape[1]:
    #         i = im.shape[1]
    xs[xs > im.shape[1]] = im.shape[1]

    # for j in ys:
    #     if j > im.shape[0]:
    #         j = im.shape[0]
    ys[ys > im.shape[0]] = im.shape[0]

    xs = xs.astype(np.int)
    ys = ys.astype(np.int)
    xs_min = xs.min()
    ys_min = ys.min()
    szx = sz[0]
    szy = sz[1]
    if ys[-1] + 1 - ys[0] == 190:
        out = im[ys[0] - 1:ys[-1], xs[0] - 1:xs[-1]]
        print('good')
    else:
        out = im[ys_min:(ys_min + szx), xs_min:(xs_min + szy)]
        print('bad')
    out = out - np.mean(out)
    out = window * out

    return out
