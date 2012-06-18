import cv2
from SimpleCV import *
import numpy as np
from math import floor
from cv2 import *
from cv2.cv import *
import scipy.spatial.distance as spsd
import math
import time
from random import shuffle, random

class VarianceFilter:
    integralImage = []
    integralImage_squared = []
    enabled = False
    windowOffsets = []
    minVar = 0.0
    
    def __init__(self):
        self.enabled = True
        self.minVar = 0
        self.integralImage = None
        self.integralImage_squared = None
        
    def calcVariance(self, off):
        ii1 = self.integralImage.data
        ii2 = self.integralImage_squared.data
        
        mX  = (ii1[off[3]] - ii1[off[2]] - ii1[off[1]] + ii1[off[0]]) / float(off[5])
        mX2 = (ii2[off[3]] - ii2[off[2]] - ii2[off[1]] + ii2[off[0]]) / float(off[5])
        return mX2 - mX*mX;
        
    def nextIteration(img):
        if not self.enabled:
            return
        self.integralImage = IntegralImage(img.size())
        self.integralImage.calcIntImg(img)
        
        self.integralImage_squared = IntegralImage(img.size())
        integralImg_squared.calcIntImg(img, True)
        
    def filter(self, i):
        if not self.enabled:
            return True

        bboxvar = self.calcVariance(self.windowOffsets[TLD_WINDOW_OFFSET_SIZE*i:])
        self.detectionResult.variances[i] = bboxvar;

        if bboxvar < minVar:
            return False

        return True
        
    def release(self):
        self.integralImage = None
        self.integralImage_squared = None

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
    return img[boundary]
    
def tldExtractNormalizedPatchBB(img, boundary):
    x,y,w,h = tldExtractDimsFromArray(boundary)
    output = tldExtractNormalizedPatch(img, x,y,w,h)
    return output
    
def tldExtractNormalizedPatchRect(img, rect):
    output = tldExtractNormalizedPatch(img, rect[0],rect[1],rect[2],rect[3])
    
def calculateMean(value):
    return np.array(value).mean()

def tldCalcVariance(value):
    return np.array(value).std()
    
def tldBBOverlap(bb1, bb2):
    if bb1[0] > bb2[0]+bb2[2] or bb1[1] > bb2[1]+bb2[3] or bb1[0]+bb1[2] < bb2[0] or bb1[1]+bb1[3] < bb2[1]:
        return 0.0

    colInt =  min(bb1[0]+bb1[2], bb2[0]+bb2[2]) - max(bb1[0], bb2[0]);
    rowInt =  min(bb1[1]+bb1[3], bb2[1]+bb2[3]) - max(bb1[1], bb2[1]);

    intersection = colInt * rowInt;
    area1 = bb1[2]*bb1[3];
    area2 = bb2[2]*bb2[3];
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
    return tldOverlap(windows, numWindows, bb)

def tldOverlap(windows, numWindows, boundary):
    overlap = []
    for i in range(numWindows):
        overlap.append(tldBBOverlap(boundary, windows[TLD_WINDOW_SIZE*i:]))
    return overlap
    
def tldSortByOverlapDesc(bb1 , bb2):
    return bb1[1], bb2[1]
    
def tldIsInside(bb1, bb2):
    if bb1[0] > bb2[0] and bb1[1] > bb2[1] and bb1[0]+bb1[2] < bb2[0]+bb2[2] and bb1[1]+bb1[3] < bb2[1]+bb2[3]:
        return 1
    else: 
        return 0

TLD_PATCH_SIZE = 15

class NNClassifier:
    enabled = False
    windows = []
    thetaFP = 0.0
    thetaTP = 0.0
    detectionResult = DetectionResult()
    falsePositives = []
    truePositives = []
    
    def __init__(self):
        self.thetaFP = 0.5
        self.thetaTP = 0.65
        
    def ncc(self, f1, f2):
        corr = 0
        norm1 = 0
        norm2 = 0
        
        size = TLD_PATCH_SIZE*TLD_PATCH_SIZE
        
        for i in range(size):
            corr += f1[i]*f2[i]
            norm1 += f1[i]*f1[i]
            norm2 += f2[i]*f2[i]
            
        return (corr / (norm1*norm2)**0.5 + 1) / 2.0
        
    def classifyPatch(self, patch):
        if not self.truePositives:
            return 0
        if not self.falsePositives:
            return 0
        
        corr_maxp = 0
        for i in range(len(self.truePositives)):
            ccorr = ncc(self.truePositives[i].values, patch.values)
            if ccorr > ccorr_max_p:
                ccorr_max_p = ccorr
        
        ccorr_max_n = 0
        for i in range(len(self.falsePositives)):
            ccorr = ncc(self.falsePositives[i].values, patch.values)
            if ccorr > ccorr_max_n:
                ccorr_max_n = ccorr
        
        dN = 1-ccorr_max_n
        dP = 1-ccorr_max_p
        distance = float(dN)/(dN+dP)
        
        return distance
        
    def classifyBB(self, img, bb):
        patch = NormalizedPatch()
        bbox = self.windows[TLD_WINDOW_SIZE*windowIdx:]
        patch.values = tldExtractNormalizedPatchBB(img, bbox)
        return self.classifyPatch(patch)
        
    def filter(self, img, windowIdx):
        if not self.enabled:
            return True
        conf = self.classifyWindow(img, windowIdx)
        if conf < self.thetaTP:
            return False
        return True
        
    def learn(self, patches):
        for i in range(len(patches)):
            patch = patches[i]
            conf = classifyPatch(patch)
            
            if patch.positive and conf < self.thetaTP:
                self.truePositives.append(patch)
            if not patch.positive and conf >= self.thetaFP:
                self.falsePositives.append(patch)
                
    def release(self):
        self.truePositives = []
        self.falsePositives = []
      
class NormalizedPatch:
    values = []
    positive = False
    
    def __init__(self):
        values = [0.0]*(TLD_PATCH_SIZR,TLD_PATCH_SIZE)
        positive = False

class MedianFlowTracker:
    trackerBB = []
    
    def __init__(self):
        pass
    
    def track(self, prevImg, currImg, prevBB):
        # prevBB -> x1,y1,w,h
        # bb_tracker -> x1,y1,x2,y2
        if prevBB:
            if prevBB[2] <= 0 or prevBB[3] <= 0 :
                return
            bb_tracker = [prevBB[0], prevBB[1], prevBB[0]+prevBB[2]-1, prevBB[1]+prevBB[3]-1]
            
            bb_tracker, shift = fbtrack(prevImg, currImg, bb_tracker)
            
            x = floor(bb_tracker[0]+0.5)
            y = floor(bb_tracker[1]+0.5)
            w = floor(bb_tracker[2]-bb_tracker[0]+1+0.5)
            h = floor(bb_tracker[3]-bb_tracker[1]+1+0.5)
            
            if x<0 or y<0 or w<=0 or h<=0 or x +w > currImg.cols or y+h > currImg.rows or not x or not y or not w or not h:
                pass
            else:
                self.trackerBB = [x,y,w,h]

class IntergralImage:
    data = []
    width = 0
    height = 0
    
    def __init__(size):
        self.data = [0.0]*(size[0]*size[1])
        
    def calcIntImage(img, squared=False):
        output = self.data
        ip = img.data
        for i in range(img.height):
            for j in range(img.width):
                A = 0
                if i >0:
                    A = output[img.height *j+i-1]
                B = 0
                if j>0:
                    B = output[img.height*(j-1)+i]
                C = 0
                if i>0 and j>0:
                    C = output[img.height*(j-1)+i-1]
                value = ip[img.step*j+i]
                if squared:
                    value = value**2
                output[img.height*j+i] = A+B-C+value
        self.data = output

class ForegroundDetector:
    
    fgThreshold = 16
    minBlobSize = 0
    bgImg = None
    detectionResult = DetectionResult()
    
    def __init__(self, fgThreshold=16, minBlobSize=0):
        self.fgThreshold = fgThreshold
        self.minBlobSize = minBlobSize
        #self.bgImg = bgImg
    
    def nextIteration(self, img):
        absImg = cv2.absdiff(self.bgImg, img)
        threshImg = cv2.threshold(absImg,self.fgThreshold,255,cv2.THRESH_BINARY)
        im = threshImg
        #//blobs = CBlobResult(im, None, 0)
        #//blobs.Filter( blobs, B_EXCLUDE, CBlobGetArea(), B_LESS, minBlobSize )
        fgList = self.detectionResult.fgList
        fgList = []
        
        for i in range(blobs.getNumBlobs()):
            #//blob = blobs.GetBlob(i)
            #//rect = blob.GetBoundingBox()
            fgList.append(rect)
        
    def isActive(self):
        return (not self.bgImg)
        
    def release(self):
        pass

def sub2idx(x,y,widthstep):
    return (int(floor((x)+0.5) + floor((y)+0.5)*(widthstep)))

class EnsembleClassifier:
    img = ''
    enabled = True
    numTrees = 10
    numFeatures = 13
    imgWidthStep = 0
    numScales = 0
    scales = []
    windowOffsets = []
    featureOffsets = []
    features = []
    numIndices = 0
    posteriors = []
    positives = []
    negatives = []
    detectionResult = DetectionResult()
    
    def __init__(self):
        pass

    def init(self):
        self.numIndices = pow(2.0,self.numFeatures)
        self.initFeatureLocations()
        self.initFeatureOffsets()
        self.initPosteriors()
        
    def release(self):
        self.features = None
        self.featureOffsets = None
        self.posteriors = None
        self.positives = None
        self.negatives = None
        
    def initFeatureLocations(self):
        size = 2 * 2 * numFeatures * numTrees
        self.features = []
        for i in range(size):
            self.features.append(random())
            
    def initFeatureOffsets(self):
        off = []
        for k in range(self.numScales):
            scale = self.scales[k]
            for i in range(self.numTrees):
                for j in range(self.numFeatures):
                    currentFeature = self.features[4*self.numFeatures*i+4*j:]
                    off.append(sub2idx((scale[0]-1)*currentFeature[0]+1,(scale[1]-1)*currentFeature[1]+1,self.imgWidthStep))
                    off.append(sub2idx((scale[0]-1)*currentFeature[2]+1,(scale[1]-1)*currentFeature[3]+1,self.imgWidthStep))
        self.featureOffsets[:len(off)] = off
                    
    def initPosteriors(self):
        self.posteriors = [None]*(self.numTrees*self.numIndices)
        self.positives = [None]*(self.numTrees*self.numIndices)
        self.negatives = [None]*(self.numTrees*self.numIndices)
        
        for i in range(self.numTrees):
            for j in range(self.numIndices):
                self.posteriors[i*self.numIndices+j]=0
                self.positives[i*self.numIndices+j]=0
                self.negatives[i*self.numIndices+j]=0
                
    def nextIteration(self,img):
        self.img = img.data
        
    def calcFernFeature(self, windowIdx, treeIdx):
        index = 0
        bbox = self.windowOffsets[windowIdx+TLD_WINDOW_OFFSET_SIZE:]
        off = self.featureOffsets[bbox[4]+treeIdx*2*numFeatures:]
        for i in range(self.numFeatures):
            index <<= 1
            fp0 = img[bbox[0]+off[0]]
            fp1 = img[bbox[0]+off[1]]
            if fp0 > fp1:
                index |= 1
            off = off[2:]
        return index
        
    def calcFeatureVector(self, windowIdx):
        featureVector = []
        for i in range(self.numTrees):
            featureVector.append(self.calcFernFeature(windowIdx, i))
        return featureVector
        
    def calcConfidence(self, featureVector):
        conf = 0.0
        for i in range(self.numTrees):
            conf += posteriors[i * numIndices + featureVector[i]]
        return conf
        
    def classifyWindow(self, windowIdx):
        #featureVector = self.detectionResult.featureVectors[self.numTrees*windowIdx:]
        featureVector = self.calcFeatureVector(windowIdx)
        self.detectionResult.posteriors[windowIdx] = self.calcConfidence(featureVector)
        
    def filter(self, i):
        if not self.enabled:
            return True
        self.classifyWindow(i)
        if(self.detectionResult.posteriors[i] < 0.5):
            return False
        return True
        
    def updatePosterior(self, treeIdx, idx, positive, amount):
        index = treeIdx * numIndices + idx
        if positive:
            self.positives[index] += amount
        else:
            self.negatives[index] += amount
            
        self.posteriors[index] = float(positives[index]) / (positives[index] + negatives[index]) / 10.0
        
    def updatePosterior(self, featureVector, positive, amount):
        for i in range(self.numTrees):
            idx = featureVector[i]
            self.updatePosterior(i, idx, positive, amount)
    
    def learn(self, img, boundary, positive, featureVector):
        if not self.enabled:
            return 
        conf = self.calcConfidence(featureVector)
        if (positive and conf < 0.5) or (not positive and conf > 0.5):
            self.updatePosteriors(featureVector, positive, 1)

TLD_WINDOW_SIZE = 5;
TLD_WINDOW_OFFSET_SIZE = 6

class DetectorCascade:
    numScales = 0
    scales = []
    minScale = -10
    maxScale = 10
    useShift = 1
    shift = 0.1
    minSize = 25
    numFeatures = 10
    numTrees = 13
    imgWidth = -1
    imgHeight = -1
    imgWidthStep = -1
    objWidth = -1
    objHeight = -1
    numWindows = 0
    windows = []
    windowOffsets = []
    initialised = False
    
    foregroundDetector = ForegroundDetector()
    varianceFilter = VarianceFilter()
    ensembleClassifier = EnsembleClassifier()
    clustering = Clustering()
    nnClassifier = NNClassifier()
    
    detectionResult = DetectionResult()
    
    def __init__(self):
        pass
        
    def init(self):
        self.initWindowsAndScales()
        self.initWindowOffsets()
        self.propagateMembers()
        self.ensembleClassifier.init()
        self.initialised = True
        
    def propgateMembers(self):
        self.detectionResult.init(self.numWindows,self.numTrees)
        self.varianceFilter.windowOffsets = self.windowOffsets
        self.ensembleClassifier.windowOffsets = self.windowOffsets
        self.ensembleClassifier.imgWidthStep = self.imgWidthStep
        self.ensembleClassifier.numScales = self.numScales
        self.ensembleClassifier.scales = self.scales
        self.ensembleClassifier.numFeatures = self.numFeatures
        self.ensembleClassifier.numTrees = self.numTrees
        self.nnClassifier.windows = self.windows
        self.clustering.windows = self.windows
        self.clustering.numWindows = self.numWindows

        self.foregroundDetector.minBlobSize = self.minSize*self.minSize

        self.foregroundDetector.detectionResult = self.detectionResult
        self.varianceFilter.detectionResult = self.detectionResult
        self.ensembleClassifier.detectionResult = self.detectionResult
        self.nnClassifier.detectionResult = self.detectionResult
        self.clustering.detectionResult = self.detectionResult
    
    def initWindowsAndScales(self):
        scanAreaX = 1
        scanAreaY = 1
        scanAreaW = self.imgWidth-1
        scanAreaH = self.imgHeight-1
        scaleIndex = 0
        windowIndex = 0
        self.scales = [(0,0)]*(self.maxScale-self.minScale+1)
        self.numWindows = 0
        
        for i in range(self.minScale,self.maxScale+1):
            scale = pow(1.2,i)
            w = self.objWidth*scale
            h = self.objHeight*scale
            if self.useShift:
                ssw = max(1,w*self.shift)
                ssh = max(1,h*self.shift)
            else:
                ssw = 1
                ssh = 1
            
            if w < self.minSize or h < self.minSize or w > self.scanAreaW or h > self.scanAreaH: 
                continue
            self.scales[scaleIndex][0] = w
            self.scales[scaleIndex][1] = h
            scaleIndex+=1
            numWindows += floor(float(self.scanAreaW - w + ssw)/ssw)*floor(float(self.scanAreaH - h + ssh) / ssh)
            
        self.numScales = scaleIndex
        self.windows = [0]*(TLD_WINDOW_SIZE*numWindows)
        
        for scaleIndex in range(self.numScales):
            w = self.scales[scaleIndex][0]
            h = self.scales[scaleIndex][1]
            if self.useShift:
                ssw = max(1,w*self.shift)
                ssh = max(1,h*self.shift)
            else:
                ssw = 1
                ssh = 1
            
            y = scanAreaY
            while y + h <= scanAreaY +scanAreaH:
                x = scanAreaX
                while x + w <= scanAreaX + scanAreaW:
                    bb = self.windows[TLD_WINDOW_SIZE*windowIndex:]
                    x, y, w, h = bb[:4]
                    bb[4] = scaleIndex
                    windowIndex+=1
                    x+=ssw
                y+=ssh
        #//assert(windowIndex == numWindows)
        
    def initWindowOffsets(self):
        self.windowOffsets = [0]*TLD_WINDOW_OFFSET_SIZE*numWindows
        off = []
        
        windowSize = TLD_WINDOW_SIZE
        for i in range(self.numWindows):
            window = self.windows[windowSize*i:]
            off.append(sub2idx(window[0]-1,window[1]-1,imgWidthStep))
            off.append(sub2idx(window[0]-1,window[1]+window[3]-1,imgWidthStep))
            off.append(sub2idx(window[0]+window[2]-1,window[1]-1,imgWidthStep))
            off.append(sub2idx(window[0]+window[2]-1,window[1]+window[3]-1,imgWidthStep))
            off.append(window[4]*2*numFeatures*numTrees)
            off.append(window[2]*window[3])
        self.windowOffsets[:len(off)]=off
        
    def detect(self, img):
        self.detectionResult.reset()
        if not self.initialised:
            return
        self.foregroundDetector.nextIteration(img)
        self.varianceFilter.nextIteration(img)
        self.ensembleClassifier.nextIteration(img)
        
        #multiprocessing stuff .. what ??
        for i in range(self.numWindows):
            window = self.windows[TLD_WINDOW_SIZE*i:]
            if self.foregroundDetector.isActive():
                inInside = False
                for j in range(len(self.detectionResult.fgList)):
                    bgBox = self.detectionResult.fgList[j:j+4]
                    if tldIsInside(window, bgBox):
                        inInside = True
                if not isInside:
                    self.detectionResult.posteriors[i] = 0
                    continue
            if not self.varianceFilter.filter(i):
                self.detectionResult.posteriors[i] = 0
                continue
            if not self.ensembleClassifier.filter(i):
                continue
            if not self.nnClassifier.filter(img,i):
                continue
            
            self.detectionResult.confidentIndices.append(i)
        
        self.clustering.clusterConfidentIndices()
        self.detectionResult.containsValidData = True
        
    def cleanPreviousData(self):
        self.detectionResult.reset()
        
    def release(self):
        if not self.initialised:
            return
        self.initialised = False
        self.foregroundDetector.release()
        self.ensembleClassifier.release()
        self.nnClassifier.release()
        self.clustering.release()
        self.numWindows = 0
        self.numScales = 0
        self.scales = []
        self.windows = []
        self.windowOffsets = []
        self.objWidth = -1
        self.objHeight = -1
        self.detectionResult.release()

class Clustering:
    windows = []
    numWindows = 0
    detectionResult = DetectionResult()
    cutoff = 0.0
    
    def __init__(self):
        self.cutoff = 0.5
        self.windows = None
        self.numWindows = 0
        
    def calcMeantRect(self, indices):
        x=y=w=h=0
        numIndices = len(indices)
        for i in range(numIndices):
            bb = windows[TLD_WINDOW_SIZE*indices[i]:]
            x += bb[0]
            y += bb[1]
            w += bb[2]
            h += bb[3]
            
        x /= numIndices
        y /= numIndices
        w /= numIndices
        h /= numIndices
        
        self.detectionResult.detectorBB = [None]*4
        self.detectionResult.detectorBB[0] = floor(x+0.5)
        self.detectionResult.detectorBB[1] = floor(y+0.5)
        self.detectionResult.detectorBB[2] = floor(w+0.5)
        self.detectionResult.detectorBB[3] = floor(h+0.5)
        
    def calcDistances(self, distances):
        confidentIndices = self.detectionResult.confidentIndices
        indices_size = len(confidentIndices)
        for i in range(indices_size):
            firstIndex = confidentIndices[0]
            confidentIndices.pop(0)
            distances_tmp = tldOverlapOne(windows, numWindows, firstIndex, confidentIndices)
            distances_tmp += indices_size-i-1
            
        for i in range(indices_size*(indices_size-1)/2):
            distances[i] = 1-distances[i]
            
        return distances
        
    def clusterConfidentIndices(self):
        numConfidentIndices = len(self.detectionResult.confidentIndices)
        distances = [0.0]*(numConfidentIndices*(numConfidentIndices-1)/2)
        distances = self.calcDistances(distances)
        clusterIndices = [0]*(numConfidentIndices)
        self.cluster(distances, clusterIndices)
        if(self.detectionResult.numClusters == 1):
            self.calcMeanRect(self.detectionResult.confidentIndices)
            
    def cluster(self, distances, clusterIndices):
        numConfidentIndices = len(self.detectionResult.confidentIndices)

        if(numConfidentIndices == 1):
            clusterIndices[0] = 0
            self.detectionResult.numClusters = 1
            return
            
        numDistances = numConfidentIndices*(numConfidentIndices-1)/2
        distUsed = [0]*numDistances
        clusterIndices = [-1]*numConfidentIndices
        
        newClusterIndex = 0;
        numClusters = 0;
        
        while(True):
            shortestDist = -1
            shortestDistIndex = -1
            distIndex = 0
            for i in range(numConfidentIndices):
                for j in range(i+1,numConfidentIndices):
                    if(not distUsed[distIndex] and (shortestDistIndex == -1 or distances[distIndex] < shortestDist)):
                        shortestDist = distances[distIndex]
                        shortestDistIndex = distIndex
                        i1=i
                        i2=j
                        
                distIndex+=1
                
            if(shortestDistIndex == -1):
                break
            
            distUsed[shortestDistIndex] = 1
            if clusterIndices[i1] == -1 and clusterIndices[i2] == -1:
                if shortestDist < cutoff:
                    clusterIndices[i1] = clusterIndices[i2] = newClusterIndex
                    newClusterIndex+=1
                    numClusters+=1
                else:
                    clusterIndices[i1] = newClusterIndex
                    newClusterIndex+=1
                    numClusters+=1
                    clusterIndices[i2] = newClusterIndex
                    newClusterIndex+=1
                    numClusters+=1
            
            elif clusterIndices[i1] == -1 and clusterIndices[i2] != -1:
                if shortestDist < cutoff:
                    clusterIndices[i1] = clusterIndices[i2]
                else:
                    clusterIndices[i1] = newClusterIndex
                    newClusterIndex+=1
                    numClusters+=1
            
            elif clusterIndices[i1] != -1 and clusterIndices[i2] == -1:
                if shortestDist < cutoff:
                    clusterIndices[i2] = clusterIndices[i1]
                else:
                    clusterIndices[i2] = newClusterIndex
                    newClusterIndex+=1
                    numClusters+=1
            
            else:
                if clusterIndices[i1] != clusterIndices[i2] and shortestDist < cutoff:
                    oldClusterIndex = clusterIndices[i2]
                    for i in range(numConfidentIndices):
                        if clusterIndices[i] == oldClusterIndex:
                            clusterIndices[i] = clusterIndices[i1]
                    numClusters -=1
                    
        self.detectionResult.numClusters = numClusters

class DetectionResult:
    containsValidData = False
    fgList = []
    confidentIndices = []
    numClusters = 0
    detectorBB = None
    variances = None
    posteriors = None
    featureVectors = None
    
    def __init__(self):
        pass
        
    def init(self, numWindows, numTrees):
        self.variances = [0.0]*numWindows
        self.posteriors = [0.0]*numWindows
        self.featureVectors = [0.0]*(numWindows*numTrees)
        self.confidentIndices = []
        
    def reset(self):
        self.containsValidData = False
        self.fgList = []
        self.confidentIndices = []
        self.numClusters = 0
        self.detectorBB = None
        
    def release(self):
        self.fgList = []
        self.variances = None
        self.posteriors = None
        self.featureVectors = None
        self.detectorBB = None
        self.containsValidData = None

def calculateBBCenter(bb):
    """
    
    **SUMMARY**
    
    Calculates the center of the given bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    center - A tuple of two floating points
    
    """
    center = (0.5*(bb[0] + bb[2]),0.5*(bb[1]+bb[3]))
    return center
    
def getFilledBBPoints(bb, numM, numN, margin):
    """
    
    **SUMMARY**
    
    Creates numM x numN points grid on Bounding Box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    numM - Number of points in height direction.
    numN - Number of points in width direction.
    margin - margin (in pixel)
    
    **RETURNS**
    
    pt - A list of points (pt[0] - x1, pt[1] - y1, pt[2] - x2, ..)
    
    """
    pointDim = 2
    bb_local = (bb[0] + margin, bb[1] + margin, bb[2] - margin, bb[3] - margin)
    if numM == 1 and numN == 1 :
        pts = calculateBBCenter(bb_local)
        return pts
    
    elif numM > 1 and numN == 1:
        divM = numM - 1
        divN = 2
        spaceM = (bb_local[3]-bb_local[1])/divM
        center = calculateBBCenter(bb_local)
        pt = [0.0]*(2*numM*numN)
        for i in range(numN):
            for j in range(numM):
                pt[i * numM * pointDim + j * pointDim + 0] = center[0]
                pt[i * numM * pointDim + j * pointDim + 1] = bb_local[1] + j * spaceM
                
        return pt
        
    elif numM == 1 and numN > 1:
        divM = 2
        divN = numN - 1
        spaceN = (bb_local[2] - bb_local[0]) / divN
        center = calculateBBCenter(bb_local)
        pt = [0.0]*((numN-1)*numM*pointDim+numN*pointDim)
        for i in range(numN):
            for j in range(numN):
                pt[i * numM * pointDim + j * pointDim + 0] = bb_local[0] + i * spaceN
                pt[i * numM * pointDim + j * pointDim + 1] = center[1]
        return pt
        
    elif numM > 1 and numN > 1:
        divM = numM - 1
        divN = numN - 1
    
    spaceN = (bb_local[2] - bb_local[0]) / divN
    spaceM = (bb_local[3] - bb_local[1]) / divM

    pt = [0.0]*((numN-1)*numM*pointDim+numM*pointDim)
    
    for i in range(numN):
        for j in range(numM):
            pt[i * numM * pointDim + j * pointDim + 0] = float(bb_local[0] + i * spaceN)
            pt[i * numM * pointDim + j * pointDim + 1] = float(bb_local[1] + j * spaceM)
    return pt

def getBBWidth(bb):
    """
    
    **SUMMARY**
    
    Get width of the bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    width of the bounding box
    
    """
    return bb[2]-bb[0]+1
    
def getBBHeight(bb):
    """
    
    **SUMMARY**
    
    Get height of the bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    height of the bounding box
    
    """
    return bb[3]-bb[1]+1
    
def predictBB(bb0, pt0, pt1, nPts):
    """
    
    **SUMMARY**
    
    Calculates the new (moved and resized) Bounding box.
    Calculation based on all relative distance changes of all points
    to every point. Then the Median of the relative Values is used.
    
    **PARAMETERS**
    
    bb0 - Bounding Box represented through 2 points (x1,y1,x2,y2)
    pt0 - Starting Points
    pt1 - Target Points
    nPts - Total number of points (eg. len(pt0))
    
    **RETURNS**
    
    bb1 - new bounding box
    shift - relative scale change of bb0
    
    """
    ofx = []
    ofy = []
    for i in range(nPts):
        ofx.append(pt1[i][0]-pt0[i][0])
        ofy.append(pt1[i][1]-pt0[i][1])
    
    dx = getMedianUnmanaged(ofx)
    dy = getMedianUnmanaged(ofy)
    ofx=ofy=0
    
    lenPdist = nPts * (nPts - 1) / 2
    dist0=[]
    for i in range(nPts):
        for j in range(i+1,nPts):
            temp0 = ((pt0[i][0] - pt0[j][0])**2 + (pt0[i][1] - pt0[j][1])**2)**0.5
            temp1 = ((pt1[i][0] - pt1[j][0])**2 + (pt1[i][1] - pt1[j][1])**2)**0.5
            dist0.append(float(temp1)/temp0)
            
    shift = getMedianUnmanaged(dist0)
    
    s0 = 0.5 * (shift - 1) * getBBWidth(bb0)
    s1 = 0.5 * (shift - 1) * getBBHeight(bb0)
    
    bb1 = (abs(bb0[0] + s0 + dx),
           abs(bb0[1] + s1 + dy),
           abs(bb0[2] + s0 + dx), 
           abs(bb0[3] + s1 + dy))
              
    return (bb1, shift)
    
def getBB(pt0,pt1):
    xmax = np.max((pt0[0],pt1[0]))
    xmin = np.min((pt0[0],pt1[0]))
    ymax = np.max((pt0[1],pt1[1]))
    ymin = np.min((pt0[1],pt1[1]))
    return xmin,ymin,xmax,ymax
    
def getRectFromBB(bb):
    return bb[0],bb[1],bb[0]-bb[2],bb[1]-bb[3]
    
def fbtrack(imgI, imgJ, bb, numM=10, numN=10,margin=5,winsize_ncc=10):
    """
    **SUMMARY**
    
    Forward-Backward tracking using Lucas-Kanade Tracker
    
    **PARAMETERS**
    
    imgI - Image contain Object with known BoundingBox (Numpy array)
    imgJ - Following image (Numpy array)
    bb - Bounding box represented through 2 points (x1,y1,x2,y2)
    numM - Number of points in height direction.
    numN - Number of points in width direction.
    margin - margin (in pixel)
    winsize_ncc - size of the search window at each pyramid level in LK tracker (in int)
    
    **RETURNS**
    
    newbb - Bounding box of object in track in imgJ
    scaleshift - relative scale change of bb
    
    """
    nPoints = numM*numN
    sizePointsArray = nPoints*2
    
    pt = getFilledBBPoints(bb, numM, numN, margin)
    fb, ncc, status, ptTracked = lktrack(imgI, imgJ, pt, nPoints, winsize_ncc)
    
    nlkPoints = 0
    for i in range(nPoints):
        nlkPoints += status[i][0]

    startPoints = []
    targetPoints = []
    fbLKCleaned = [0.0]*nlkPoints
    nccLKCleaned = [0.0]*nlkPoints
    M = 2
    nRealPoints = 0
    
    for i in range(nPoints):
        if ptTracked[M*i] is not None:
            startPoints.append((pt[2 * i],pt[2*i+1]))
            targetPoints.append((ptTracked[2 * i], ptTracked[2 * i + 1]))
            fbLKCleaned[nRealPoints]=fb[i]
            nccLKCleaned[nRealPoints]=ncc[i][0][0]
            nRealPoints+=1
            
    medFb = getMedian(fbLKCleaned)
    medNcc = getMedian(nccLKCleaned)
    
    nAfterFbUsage = 0
    for i in range(nlkPoints):
        if fbLKCleaned[i] <= medFb and nccLKCleaned[i] >= medNcc:
            startPoints[nAfterFbUsage] = startPoints[i]
            targetPoints[nAfterFbUsage] = targetPoints[i]
            nAfterFbUsage+=1

    newBB, scaleshift = predictBB(bb, startPoints, targetPoints, nAfterFbUsage)
    
    return (newBB, scaleshift)

def lktrack(img1, img2, ptsI, nPtsI, winsize_ncc=10, win_size_lk=4, method=CV_TM_CCOEFF_NORMED):
    """
    **SUMMARY**
    
    Lucas-Kanede Tracker with pyramids
    
    **PARAMETERS**
    
    img1 - Previous image or image containing the known bounding box (Numpy array)
    img2 - Current image
    ptsI - Points to track from the first image
           Format ptsI[0] - x1, ptsI[1] - y1, ptsI[2] - x2, ..
    nPtsI - Number of points to track from the first image
    winsize_ncc - size of the search window at each pyramid level in LK tracker (in int)
    method - Paramete specifying the comparison method for normalized cross correlation 
             (see http://opencv.itseez.com/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#cv2.matchTemplate)
    
    **RETURNS**
    
    fb - forward-backward confidence value. (corresponds to euclidean distance between).
    ncc - normCrossCorrelation values
    status - Indicates positive tracks. 1 = PosTrack 0 = NegTrack
    ptsJ - Calculated Points of second image
    
    """
    template_pt = []
    target_pt = []
    fb_pt = []
    ptsJ = [0.0]*len(ptsI)
    
    for i in range(nPtsI):
        template_pt.append((ptsI[2*i],ptsI[2*i+1]))
        target_pt.append((ptsI[2*i],ptsI[2*i+1]))
        fb_pt.append((ptsI[2*i],ptsI[2*i+1]))
    
    template_pt = np.asarray(template_pt,dtype="float32")
    target_pt = np.asarray(target_pt,dtype="float32")
    fb_pt = np.asarray(fb_pt,dtype="float32")
    
    target_pt, status, track_error = calcOpticalFlowPyrLK(img1, img2, template_pt, target_pt, 
                                     winSize=(win_size_lk, win_size_lk), flags = OPTFLOW_USE_INITIAL_FLOW,
                                     criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))
                                     
    fb_pt, status_bt, track_error_bt = calcOpticalFlowPyrLK(img2,img1, target_pt,fb_pt, 
                                       winSize = (win_size_lk,win_size_lk),flags = OPTFLOW_USE_INITIAL_FLOW,
                                       criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))
    
    for i in range(nPtsI):
        if status[i] == 1 and status_bt[i] == 1:
            status[i]=1
        else:
            status[i]=0
    ncc = normCrossCorrelation(img1, img2, template_pt, target_pt, status, winsize_ncc, method)
    fb = euclideanDistance(template_pt, target_pt)
    
    for i in range(nPtsI):
        if status[i] == 1 and status_bt[i] == 1:
            ptsJ[2 * i] = target_pt[i][0]
            ptsJ[2 * i + 1] = target_pt[i][1]
        else:
            ptsJ[2 * i] = None
            ptsJ[2 * i + 1] = None
            fb[i] = None
            ncc[i] = None
            
    return fb, ncc, status, ptsJ
    
def euclideanDistance(point1,point2):
    """
    **SUMMARY**
    
    Calculates eculidean distance between two points
    
    **PARAMETERS**
    
    point1 - vector of points
    point2 - vector of points with same length
    
    **RETURNS**
    
    match = returns a vector of eculidean distance
    """
    match=[]
    n = len(point1)
    for i in range(n):
        match.append(spsd.euclidean(point1[i],point2[i]))
    return match

def normCrossCorrelation(img1, img2, pt0, pt1, status, winsize, method=CV_TM_CCOEFF_NORMED):
    """
    **SUMMARY**
    
    Calculates normalized cross correlation for every point.
    
    **PARAMETERS**
    
    img1 - Image 1.
    img2 - Image 2.
    pt0 - vector of points of img1
    pt1 - vector of points of img2
    status - Switch which point pairs should be calculated.
             if status[i] == 1 => match[i] is calculated.
             else match[i] = 0.0
    winsize- Size of quadratic area around the point
             which is compared.
    method - Specifies the way how image regions are compared. see cv2.matchTemplate
    
    **RETURNS**
    
    match - Output: Array will contain ncc values.
            0.0 if not calculated.
 
    """
    match = []
    nPts = len(pt0)
    for i in range(nPts):
        if status[i] == 1:
            patch1 = getRectSubPix(img1,(winsize,winsize),tuple(pt0[i]))
            patch2 = getRectSubPix(img2,(winsize,winsize),tuple(pt1[i]))
            match.append(matchTemplate(patch1,patch2,method))
        else:
            match.append(0.0)
    return match

def getMedianUnmanaged(a):
    low = 0
    high = len(a) - 1
    median = (low + high) / 2
    while True:
        if high < 0:
            return 0
        if high <= low:
            return a[median]
        if high == low + 1:
            if a[low] > a[high]:
                a[low],a[high] = a[high],a[low]
            return a[median]
        middle = (low+high)/2
        if a[middle] > a[high]:
            a[middle],a[high] = a[high],a[middle]
        if a[low] > a[high]:
            a[low],a[high] = a[high],a[low]
        if a[middle] > a[low]:
            a[middle],a[low] = a[low],a[middle]
        a[middle],a[low+1] = a[low+1],a[middle]
        
        ll = low + 1
        hh = high
        
        while True:
            while True:
                ll +=1
                try:
                    if a[low] > a[ll]:
                        break                    
                except IndexError:
                    break
            while True:
                hh -= 1
                try:
                    if a[hh] > a[low]:
                        break
                except IndexError:
                    break
            if hh < ll:
                break
            try:
                a[ll],a[hh] = a[hh],a[ll]
            except IndexError:
                break
        try:
            a[low],a[hh] = a[hh],a[low]
        except IndexError:
            break
        if hh <= median:
            low = ll
        if hh >= median:
            high = hh - 1

def getMedian(a):
    median = getMedianUnmanaged(a)
    return median

class TLD:
    trackEnabled = False
    detectorEnabled = False
    learningEnabled = False
    alternating = False
    
    detectorCascade = DetectorCascade()
    nnClassifier = NNClassifier()
    medianFlowTracker = MedianFlowTracker()
    
    valid = False
    wasValid = False
    prevImg = None
    currImg = None
    prevBB = None
    currBB = None
    
    currConf = 0.0
    learning = False
    
    def __init__(self):
        self.trackEnabled = True
        self.detectorEnabled = True
        self.learningEnabled = True
        self.alternating = False
        self.valid = False
        self.wasValid = False
        self.learning = False
        self.currBB = None
        
        self.nnClassifier = self.detectorCascade.nnClassifier
        
    def storeCurrentData(self):
        self.prevImg = self.currImg
        self.prevBB = self.currBB
        self.detectorCascade.cleanPreviousData()
        self.medianFlowTracker.cleanPreviousData()
        self.wasValid = self.valid
        
    def selectObject(self, img, bb):
        self.detectorCascade.release()
        self.detectorCascade.width = bb[2]
        self.detectorCascade.height = bb[3]
        self.detectorCascade.init()
        
        self.currImg = img
        self.currBB = bb
        self.currConf = 1
        self.valid = True
        self.initialLearning()
        
    def ProcessImage(self, img):
        self.storeCurrentData()
        #//cvtColor( img,grey_frame, CV_RGB2GRAY );
        #//currImg = grey_frame; // Store new image , right after storeCurrentData();
        if self.trackEnabled:
            self.medianFlowTracker.track(prevImg, currImg, prevBB)
            
        if self.detectorEnabled and (not alternating or self.medianFlowTracker.trackerBB == None):
            self.detectorCascade.detect(grey_frame)
            
        self.fuseHypotheses()
        self.learn()
        
    def fuseHypotheses(self):
        trackerBB = self.medianFlowTracker.trackerBB
        numClusters = self.detectorCascade.detectionResult.numClusters
        detectorBB = detectorCascade.detectionResult.detectorBB
        
        self.currBB = None
        self.currConf = 0
        self.valid = False
        
        confDetector = 0
        
        if numClusters == 1:
            confDetector = self.nnClassifier.classifyBB(currImg, detectorBB)
            
        if trackerBB:
            confTracker = self.nnClassifier.classifyBB(currImg, trackerBB)
            
            if numClusters == 1 and confDetector > confTracker and tldOverlapRectRect(trackerBB, detectorBB) < 0.5:
                self.currBB = detectorBB
                self.currConf = confDetector
            else:
                self.currBB = trackerBB
                self.currConf = confTracker
                if confTracker > self.nnClassifier.thetaTP:
                    self.valid = True
                elif self.wasValid and confTracker > self.nnClassifier.thetaFP:
                    self.valid = True
        
        elif numClusters == 1:
            self.currBB = detectorBB
            self.currConf = confDetector
            
    def initialLearning(self):
        self.learning = True
        detectionResult = self.detectorCascade.detectionResult
        self.detectorCascade.detect(self.currImg)
        
        patch = NormalizedPatch()
        patch.values = tldExtractNormalizedPatchRect(self.currImg, self.currBB)
        patch.positive = 1
        
        initVar = tldCalcVariance(patch.values)
        self.detectorCascade.varianceFilter.minVar = initVar/2
        
        overlap = tldOverlapRect(self.detectorCascade.windows, self.detectorCascade.numWindows, currBB)
        
        positiveIndices = []
        negativeIndices = []
        
        for i in range(self.detectorCascade.numWindows):
            if overlap[i] > 0.6:
                positiveIndices.append((i,overlap[i]))
            if overlap[i] < 0.2:
                variance = self.detectionResult.variances[i]
                if not self.detectorCascade.varianceFilter.enabled or variance > self.detectorCascade.varianceFilter.minVar:
                    negativeIndices.append(i)
        
        positiveIndices.sort(key=lambda p:p[1])
        
        patches = []
        patches.append(patch)
        
        numIterations = min(len(positiveIndices), 10)
        for i in range(numIterations):
            idx = positiveIndices[i][0]
            self.detectorCascade.ensembleClassifier.learn(self.currImg, 
                         self.detectorCascade.windows[TLD_WINDOW_SIZE*idx:], 
                         True, 
                         self.detectionResult.featureVectors[self.detectorCascade.numTrees*idx:])
        
        shuffle(negativeIndices)
        for i in range(min(100,len(negativeIndices))):
            idx = negativeIndices[i]
            patch = NormalizedPatch()
            patch.values = tldExtractNormalizedPatchBB(currImg, self.detectorCascade.windows[TLD_WINDOW_SIZE*idx:])
            patches.append(patch)
        
        self.detectorCascade.nnClassifier.learn(patches)
        
    def learn(self):
        if not self.learningEnabled or not self.valid or not self.detectorEnabled:
            self.learning = False
            return
        self.learning = True
        
        detectionResult = self.detectorCascade.detectionResult
        if not detectionResult.containsValidData:
            self.detectorCascade.detect(self.currImg)
        
        patch = NormalizedPatch()
        patch.values = tldExtractNormalizedPatchRect(currImg, currBB)
        overlap = tldOverlapRect(self.detectorCascade.windows, self.detectorCascade.numWindows, currBB)
        
        positiveIndices = []
        negativeIndices = []
        negativeIndicesForNN = []
        
        for i in range(self.detectorCascade.numWindows):
            if overlap[i] > 0.6:
                positiveIndices.append(i,overlap[i])
            if overlap[i] < 0.2:
                if not self.detectorCascade.ensembleClassifier.enabled or self.detectionResult.posteriors[i] > 0.1:
                    negativeIndices.append(i)
                if not self.detectorCascade.ensembleClassifier.enabled or self.detectionResult.posteriors[i] > 0.5:
                    negativeIndicesForNN.append(i)
                    
        positiveIndices.sort(key = lambda p:p[1])
        patches = []
        patch.positive = 1
        patches.append(patch)
        
        numIterations = min(len(positiveIndices), 10)
        
        for i in range(len(negativeIndices)):
            idx = negativeIndices[i]
            self.detectorCascade.ensembleClassifier.learn(self.currImg, 
                            self.detectorCascade,windows[TLD_WINDOW_SIZE*idx:],
                            False, 
                            self.detectionResult.featureVectors[self.detectorCascade.numTrees*idx:])
        
        for i in range(numIterations):
            idx = positiveIndices[i][0]
            self.detectorCascade.ensembleClassifier.learn(self.currImg,
                            self.detectorCascade.windows[TLD_WINDOW_SIZE*idx:],
                            True,
                            self.detectionResult.featureVectors[self.detectorCascade.numTrees*idx:])
                            
        for i in range(len(negativeIndicesForNN)):
            idx = negativeIndiceForNN[i]
            patch = NormalizedPatch()
            patch.values = tldExtractNormalizedPatchBB(currImg, self.detectorCascade.windows[TLD_WINDOW_SIZE*idx:])
            patch.positive = 0
            patches.append(patch)
            
        self.detectorCascade.nnClassifier.learn(patches)
        
class PyOpenTLD:
    tld = TLD()
    display = Display()
    threshold = 0
    initialBB = []
    
    def __init__(self):
        self.threshold = 0.5
        self.initialBB = []
        
    def start_tld(self):
        img = getImage()
        grey = img.toGray()
        
        self.tld.detectorCascade.imgWidth = grey.width
        self.tld.detectorCascade.imgHeight = grey.height
        self.tld.detectorCascade.imgWidthStep = grey.widthStep
        
        bb = getBBFromUser()
        self.tld.selectObject(grey, bb)
        skipProcessingOnce = True
        reuseFrameOnce = True
        
        self.tld.processImage(img)
        if self.tld.currBB:
            img.drawBB(self.tld.currBB)

pyOpenTLD()
