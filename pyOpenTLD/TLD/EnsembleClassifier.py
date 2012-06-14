import cv2
from random import random
from math import floor
TLD_WINDOW_OFFSET_SIZE = 10

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
    
    from DetectionResult import DetectionResult
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
        size = 2 * 2 * self.numFeatures * self.numTrees
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
        off = self.featureOffsets[bbox[4]+treeIdx*2*self.numFeatures:]
        for i in range(self.numFeatures):
            index <<= 1
            fp0 = self.img[bbox[0]+off[0]]
            fp1 = self.img[bbox[0]+off[1]]
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
            conf += self.posteriors[i * self.numIndices + featureVector[i]]
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
        index = treeIdx * self.numIndices + idx
        if positive:
            self.positives[index] += amount
        else:
            self.negatives[index] += amount
            
        self.posteriors[index] = float(self.positives[index]) / (self.positives[index] + self.negatives[index]) / 10.0
        
    def updatePosteriors(self, featureVector, positive, amount):
        for i in range(self.numTrees):
            idx = featureVector[i]
            self.updatePosterior(i, idx, positive, amount)
    
    def learn(self, img, boundary, positive, featureVector):
        if not self.enabled:
            return 
        conf = self.calcConfidence(featureVector)
        if (positive and conf < 0.5) or (not positive and conf > 0.5):
            self.updatePosteriors(featureVector, positive, 1)
