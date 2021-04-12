import numpy as np
from PIL import Image, ImageOps
import cv2 as cv

f1 = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype='float')
f1 *= 1.0/16
f2 = np.array([[1, 4, 7, 4, 1],[4, 16, 26, 16, 4],[7, 26, 41, 26, 7],
                         [4, 16, 26, 16, 4],[1, 4, 7, 4, 1]], dtype='float')
f2 *= 1.0/273

dogx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
dogy = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])

def applyFilter(filename, filter, padding):
    with Image.open(filename) as image:
        im = np.array(ImageOps.grayscale(image))

    sz = im.shape
    fz = filter.shape[0]

    im = np.pad(im, padding)

    out = np.zeros(im.shape)

    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            z = np.multiply(filter, im[i:i+fz, j:j+fz])
            out[i][j] = np.sum(z)


    img = Image.fromarray(out)
    img.show()
    return out

def sobel(filename):
    with Image.open(filename) as image:
        im = np.array(ImageOps.grayscale(image))

    sz = im.shape
    fz = dogx.shape[0]

    im = np.pad(im, 1)

    out = np.zeros(im.shape)

    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            x = np.square(np.sum(np.multiply(im[i:i+fz, j:j+fz], dogx)))
            y = np.square(np.sum(np.multiply(im[i:i+fz, j:j+fz], dogy)))
            out[i][j] = np.sqrt(x + y)

    img = Image.fromarray(out)
    img.show()
    return out


def keypoints(filename, idx):
    img = cv.imread(filename)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(img)

    img = cv.drawKeypoints(grey, kp, img)
    cv.imwrite('sift_keypoints_{}.jpg'.format(idx),img)
    
    return kp

def imgmatch(f1, f2):
    k1 = keypoints(f1, 1)
    k2 = keypoints(f2, 2)


#applyFilter('filter1_img.jpg', f1, 1)
#applyFilter('filter2_img.jpg', f1, 1)
#applyFilter('filter2_img.jpg', f2, 2)
#applyFilter('filter2_img.jpg', f2, 2)

#applyFilter('filter1_img.jpg', dogx, 1)
#applyFilter('filter1_img.jpg', dogy, 1)
#applyFilter('filter2_img.jpg', dogx, 1)
#applyFilter('filter2_img.jpg', dogy, 1)

#sobel('filter1_img.jpg')
#sobel('filter2_img.jpg')

imgmatch('SIFT1_img.jpg', 'SIFT2_img.jpg')