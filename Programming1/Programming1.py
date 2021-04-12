import numpy as np
from PIL import Image

f1 = np.array(1.0/16 * [[1, 2, 1],[2, 4, 2],[1, 2, 1]])
f2 = np.array(1.0/273 * [[1, 4, 7, 4, 1],[4, 16, 26, 16, 4],[7, 26, 41, 26, 7],
                         [4, 16, 26, 16, 4],[1, 4, 7, 4, 1]])


def applyFilter(filename, filter):
    with Image.open(filename) as im:
        ar = np.array(im)

    sz = ar.shape
    fz = filter.shape[0]

    out = np.array(ar.shape)

    for i in range(1, sz[0]-fz):
        for j in range(1, sz[1]-fz):
            out[i, j] = np.multiply(filter, ar[i:i+fz, j:j+fz])

    return out

applyFilter('filter_img.jpg', f1)