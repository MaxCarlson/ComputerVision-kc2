import numpy as np
from PIL import Image, ImageOps
import cv2 as cv
import matplotlib.pyplot as plt

import copy
import math
import random
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

f1 = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype='float')
f1 *= 1.0/16
f2 = np.array([[1, 4, 7, 4, 1],[4, 16, 26, 16, 4],[7, 26, 41, 26, 7],
                         [4, 16, 26, 16, 4],[1, 4, 7, 4, 1]], dtype='float')
f2 *= 1.0/273

dogx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
dogy = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])

def resizeImage(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)

def applyFilter(filename, filter, padding, filtername):
    with Image.open(filename) as image:
        im = np.array(ImageOps.grayscale(image))

    fz = filter.shape[0]
    out = np.zeros(im.shape)

    im = np.pad(im, padding)
    sz = im.shape


    for i in range(0, sz[0] - padding * 2):
        for j in range(0, sz[1] - padding * 2):
            z = np.multiply(filter, im[i:i+fz, j:j+fz])
            out[i][j] = np.sum(z)


    img = Image.fromarray(out)
    cv.imwrite(filename[:-4] + '_' + filtername + '.jpg',out)
    #img.show()
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
    cv.imwrite(filename[:-4] + '_sobel.jpg',out)
    #img.show()
    return out


def keypoints(filename, idx):
    img = cv.imread(filename)
    if idx == 2:
        # resize image
        img = resizeImage(img, 40)

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)

    img = cv.drawKeypoints(img, kp, img)
    cv.imwrite('sift_keypoints_{}.jpg'.format(idx),img)
    
    return img, kp, desc

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
    
    # Used to verify my 2norms and points were correct
    #matchesCV = matchesCV[:int(len(matchesCV)/10)]

    # Need to build cv.DMatch objects for simple plotting. 
    matches = []
    for i in range(len(pts)):
        matches.append(cv.DMatch(_queryIdx=int(matchesCV[i].queryIdx), _imgIdx=int(pts[i, 0]), 
                  _trainIdx=int(pts[i, 1]), _distance=pts[i, 2]))

    img3 = cv.drawMatches(im1, k1, im2, k2, matches, im2, flags=2)
    cv.imwrite('matches.jpg', img3)
    #cv.imshow('image', img3)
    #cv.waitKey(0)

class Base:
    def __init__(self, data, clusters, r, dims):
        self.r = r
        #random.seed(2)
        self.dims = dims
        self.data = data
        self.clusters = clusters
        self.maxs = []
        self.mins = []
        for d in range(self.dims):
            self.maxs.append(np.max(data[:,d]))
            self.mins.append(np.min(data[:,d]))

    def randomPoints(self):
        return [[random.uniform(min, max) for min, max in zip(self.mins, self.maxs)] 
                for _ in range(self.clusters)]


class K_Means(Base):
    def __init__(self, data, k, r, filename=None, dims=2, shape=None):
        super().__init__(data, k, r, dims)

        best = np.inf
        bestCentroids = None
        for i in range(r):
            centroids, sses, closestsCentroids = self.run()
            if sses < best:
                best = sses
                bestCentroids = np.array(centroids)[:,0,:]
        
        # 2D Plot of clusters
        if dims == 2:
            for i in range(self.clusters):
               plt.plot(self.data[closestsCentroids==i, 0], 
                        self.data[closestsCentroids==i, 1], 'o', 
                        zorder=0, color='rbgkmcy'[i % len('rbgkmcy')])
            print('BestSSE:', best)
            plt.title('SSE: %.2f' % sses + ' k='+str(k) + ' r='+str(r))
            plt.savefig(fname=filename)
            #plt.show()
            return
        
        # 3D Plot
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #for i, m in zip(range(self.clusters), 
        #                ['o', 'x', '*', '#', '@', '!', '%', '^']):
        #
        #    d = self.data[closestsCentroids==i]
        #    ax.scatter(d[:, 0], d[:, 1], d[:, 2], m)
        #
        #plt.show()

        #self.data = self.data[closestsCentroids.astype(int)] 
        #self.data = np.reshape(self.data, shape)

        # Graph pixel space
        self.data = np.take(centroids, closestsCentroids.astype(int), 0)
        self.data = np.reshape(self.data.astype('uint8'), shape)

        #self.data=resizeImage(self.data, 200)
        cv.imwrite(filename, self.data)
        #cv.imshow('image', self.data)
        #cv.waitKey(0)



    def run(self):

        centroids = self.randomPoints()
        prevBest = None
        bestCentroids = np.ones((len(self.data), 2)) * np.inf

        it = 0
        while True:
            prevBest = bestCentroids
            bestCentroids = np.ones((len(self.data), 2)) * np.inf

            # Assignment
            for i, c in enumerate(centroids): 

                # Calculate the 2 norm of all points to this centroid
                norm = np.linalg.norm((self.data - c), 2, 1)

                # Find the rows in which the 2norm is less than the previous best
                brows = np.where(norm < bestCentroids[:,1])

                # Replace the norm, then replace that best centroid index
                bestCentroids[brows, 1] = norm[brows] 
                bestCentroids[brows, 0] = i

            if (prevBest == bestCentroids).all():
                break

            # Update 
            for i, c in enumerate(centroids): 

                # Get data values belonging to centroid i
                m = np.where(bestCentroids[:,0] == i)
                x = self.data[m,:]

                # Update the centroid
                centroids[i] = 1/(np.shape(x)[1] if np.shape(x)[1] else 1) \
                   * np.sum(x, 1)
                
                vv = 0
            it += 1

        return centroids, np.sum(bestCentroids[:,1]), bestCentroids[:,0]


def runKMeans1():

    for i in [2, 3, 4]:
        data = np.genfromtxt('510_cluster_dataset.txt')
        K_Means(data, i, 10, filename='K_Means_{}'.format(i)) 

def runKMeans2():

    for i, filename in enumerate(['Kmean_img1.jpg', 'Kmean_img2.jpg']):
        for k in [5, 10]:
            data = cv.imread(filename)
            data = resizeImage(data, 100)
            shape = data.shape
            data = np.reshape(data, (data.shape[0] * data.shape[1], 3))
            K_Means(data, k, 5, dims=3, shape=shape, filename='Kmean_img{}_k={}.jpg'.format(i+1, k))

applyFilter('filter1_img.jpg', f1, 1, 'g3')
applyFilter('filter2_img.jpg', f1, 1, 'g3')
applyFilter('filter1_img.jpg', f2, 2, 'g5')
applyFilter('filter2_img.jpg', f2, 2, 'g5')

#applyFilter('filter1_img.jpg', dogx, 1, 'dogx')
#applyFilter('filter1_img.jpg', dogy, 1, 'dogy')
#applyFilter('filter2_img.jpg', dogx, 1, 'dogx')
#applyFilter('filter2_img.jpg', dogy, 1, 'dogy')
#
#sobel('filter1_img.jpg')
#sobel('filter2_img.jpg')
#
#imgmatch('SIFT1_img.jpg', 'SIFT2_img.jpg')
#runKMeans1()
#runKMeans2()