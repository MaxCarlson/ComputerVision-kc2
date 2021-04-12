import numpy as np
from PIL import Image, ImageOps
import cv2 as cv
import matplotlib.pyplot as plt

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
    if idx == 2:
        scale_percent = 40 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
  

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)

    outimg = img
    img = cv.drawKeypoints(img, kp, outimg)
    cv.imwrite('sift_keypoints_{}.jpg'.format(idx),img)
    
    return grey, kp, desc

def imgmatch(f1, f2):
    im1, k1, desc1 = keypoints(f1, 1)
    im2, k2, desc2 = keypoints(f2, 2)

    # Calculate L2 norms
    pts = np.zeros((len(k1), 3))
    for i, v in enumerate(desc1):
        pts[i, 0]   = i
        d1          = np.full(desc2.shape, v)
        n           = np.linalg.norm(d1 - desc2, 2, axis=1)
        minIdx      = np.argmin(n)
        min         = n[minIdx]
        pts[i, 1]   = minIdx
        pts[i, 2]   = min
    
    # Sort array by the minimum l2 distance between SIFT vectors
    pts = pts[pts[:,2].argsort()] 
    
    # Take lowest 10% of l2 distances
    pts = pts[0:int(len(pts)/10)]

    # Need to use one variable from these matches to build cv.DMatch
    # Objects. L2 data is not used from these
    matcher = cv.BFMatcher(cv.NORM_L2)
    matchesCV = matcher.match(desc1, desc2)
    matchesCV = sorted(matchesCV, key=lambda x: x.distance)

    # Need to build cv.DMatch objects for simple plotting. 
    matches = []
    for i in range(len(pts)):
        matches.append(cv.DMatch(_queryIdx=int(matchesCV[i].queryIdx), _imgIdx=int(pts[i, 0]), 
                  _trainIdx=int(pts[i, 1]), _distance=pts[i, 2]))

    img3 = cv.drawMatches(im1, k1, im2, k2, matches, im2, flags=2)
    plt.imshow(img3)
    plt.savefig('matches')
    plt.show()


    #l1, l2 = len(k1), len(k2)
    #pts = np.zeros((len(k1), 3))
    #
    ## Turn SIFT vectors into numpy arrays
    #kp1s = np.array([np.array(x.pt) for x in k1])
    #kp2s = np.array([np.array(x.pt) for x in k2])
    #
    ## Calculate all l2 distances between k1 and k2 SIFT vectors
    #for i, kp1 in enumerate(kp1s):
    #    pts[i, 0]   = i
    #    k1t         = np.full(kp2s.shape, kp1)
    #    n           = np.linalg.norm(k1t - kp2s, 2, axis=1)
    #    minIdx      = np.argmin(n)
    #    min         = n[minIdx]
    #    pts[i, 1]   = minIdx
    #    pts[i, 2]   = min
    #
    ## Sort array by the minimum l2 distance between SIFT vectors
    #pts = pts[pts[:,2].argsort()] 
    #
    ## Take lowest 10% of l2 distances
    #pts = pts[0:int(len(pts)/10)]


    


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