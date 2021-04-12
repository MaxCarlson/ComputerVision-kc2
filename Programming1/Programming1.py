import numpy as np
from PIL import Image

f1 = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype='float')
f1 *= 1.0/16
f2 = np.array([[1, 4, 7, 4, 1],[4, 16, 26, 16, 4],[7, 26, 41, 26, 7],
                         [4, 16, 26, 16, 4],[1, 4, 7, 4, 1]], dtype='float')
f2 *= 1.0/273

def applyFilter(filename, filter):
    with Image.open(filename) as image:
        im = np.array(image)

    sz = im.shape
    fz = filter.shape[0]

    if fz == 3:
        im = np.pad(im, 1)
    else:
        im = np.pad(im, 2)

    out = np.zeros(im.shape)

    zz = im[0:3, 0:3]

    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            z = np.matmul(filter, im[i:i+fz, j:j+fz])
            out[i][j] = np.sum(z)


    img = Image.fromarray(out)
    img.show()
    return out

applyFilter('filter1_img.jpg', f1)