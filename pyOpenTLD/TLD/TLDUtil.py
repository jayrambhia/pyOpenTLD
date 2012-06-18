import cv2
import numpy as np
from nNClassifier import NormalizedPatch

TLD_WINDOW_SIZE = 15
TLD_PATCH_SIZE = 15

def tldRectToPoints(rect):
    p1=(rect[0],rect[1])
    p2=(rect[0]+rect[2],rect[1]+rect[3])
    return p1, p2
    
def tldBoundingBoxToPoints(bb):
    return tldRectToPoints(bb)
    
def tldNormalizedImage(img):
    size = TLD_PATCH_SIZE
    result = cv2.resize(img,(size,size))
    mean = 0
    imgData = result.toString
    output = [0]*((size-1)*15+size+1)
    for i in range(15):
        for j in range(15):
            mean += imgData[j*result.step+ i]
            
    mean /= float(size*size)
    
    for i in range(size):
        for j in range(size):
            output[j*15+i] = imgData[j*result.step + i] - mean
    return output
    
def tldBoundaryToRect(boundary):
    return [boundary[0],boundary[1],boundary[2],boundary[3]]
    
def tldExtractSubImage(img, boundary):
    return img.crop(boundary[0],boundary[1],boundary[2],boundary[3])

def tldExtractNormalizedPatch(img, x, y, w, h):
    subImage = tldExtractSubImage(img, (x,y,w,h))
    return tldNormalizeImg(subImage)

def tldNormalizeImg(subimg):
    size = TLD_PATCH_SIZE;
    #result = cv2.resize(subimg,(size,size))
    result = subimg.resize(size,size)
    mean = 0.0

    imgData = result.getGrayNumpy().flat
    result = result.getMatrix()
    print "result.step",
    print result.step
    print len(imgData)

    for i in range(15):
        for j in range(15):
            mean += imgData[j*size+ i]
    mean /= size*size

    output = [0.0]*(size*15+16)

    for i in range(size):
        for j in range(size):
            output[j*15+i] = imgData[j*size + i] - mean
    return output

def tldExtractNormalizedPatchBB(img, boundary):
    x,y,w,h = tldExtractDimsFromArray(boundary)
    output = tldExtractNormalizedPatch(img, x,y,w,h)
    return output

def tldExtractDimsFromArray(boundary):
    return boundary[0],boundary[1],boundary[2],boundary[3]

def tldExtractNormalizedPatchRect(img, rect):
    output = tldExtractNormalizedPatch(img, rect[0],rect[1],rect[2],rect[3])
    return output
    
def calculateMean(value):
    return np.array(value).mean()

def tldCalcVariance(value):
    return np.array(value).std()
    
def tldBBOverlap(bb1, bb2):
    #print bb1
    #print bb2
    #if not bb2:
     #   bb2 = [0]*4
    #print "tldBBOverlap",
    if bb1[0] > bb2[0]+bb2[2] or bb1[1] > bb2[1]+bb2[3] or bb1[0]+bb1[2] < bb2[0] or bb1[1]+bb1[3] < bb2[1]:
        #print 0.0
        return 0.0

    colInt =  min(bb1[0]+bb1[2], bb2[0]+bb2[2]) - max(bb1[0], bb2[0])
    rowInt =  min(bb1[1]+bb1[3], bb2[1]+bb2[3]) - max(bb1[1], bb2[1])

    intersection = colInt * rowInt
    area1 = bb1[2]*bb1[3]
    area2 = bb2[2]*bb2[3]
    #print intersection / float(area1 + area2 - intersection)
    #print "tldBBoverlap"
    return intersection / float(area1 + area2 - intersection)

def tldOverlapOne(windows, numWindows, index, indices):
    overlap = []
    for i in range(len(indices)):
        overlap.append(tldBBOverlap(windows[TLD_WINDOW_SIZE*index:], windows[TLD_WINDOW_SIZE*indices[i]:]))
    return overlap

def tldOverlapRectRect(r1, r2):
    return tldBBOverlap(r1,r2)
    
def tldCopyRect(r):
    return r
    
def tldOverlapRect(windows, numWindows, boundary):
    return tldOverlap(windows, numWindows, boundary)

def tldOverlap(windows, numWindows, boundary):
    overlap = []
    for i in xrange(numWindows):
        index = TLD_WINDOW_SIZE*i
        bb = windows[index:index+4]
        if len(bb) < 4:
            continue
        overlap.append(tldBBOverlap(boundary, bb))
    return overlap
    
def tldSortByOverlapDesc(bb1 , bb2):
    return bb1[1], bb2[1]
    
def tldIsInside(bb1, bb2):
    if bb1[0] > bb2[0] and bb1[1] > bb2[1] and bb1[0]+bb1[2] < bb2[0]+bb2[2] and bb1[1]+bb1[3] < bb2[1]+bb2[3]:
        return 1
    else: 
        return 0
