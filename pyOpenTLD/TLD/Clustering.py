from math import floor
from TLDUtil import *
from DetectionResult import DetectionResult

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
        for i in xrange(numIndices):
            bb = self.windows[TLD_WINDOW_SIZE*indices[i]:]
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
        for i in xrange(indices_size):
            firstIndex = confidentIndices[0]
            confidentIndices.pop(0)
            distances_tmp = tldOverlapOne(self.windows, self.numWindows, firstIndex, confidentIndices)
            distances_tmp += indices_size-i-1
            
        for i in xrange(indices_size*(indices_size-1)/2):
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
            for i in xrange(numConfidentIndices):
                for j in xrange(i+1,numConfidentIndices):
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
                if shortestDist < self.cutoff:
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
                if shortestDist < self.cutoff:
                    clusterIndices[i1] = clusterIndices[i2]
                else:
                    clusterIndices[i1] = newClusterIndex
                    newClusterIndex+=1
                    numClusters+=1
            
            elif clusterIndices[i1] != -1 and clusterIndices[i2] == -1:
                if shortestDist < self.cutoff:
                    clusterIndices[i2] = clusterIndices[i1]
                else:
                    clusterIndices[i2] = newClusterIndex
                    newClusterIndex+=1
                    numClusters+=1
            
            else:
                if clusterIndices[i1] != clusterIndices[i2] and shortestDist < self.cutoff:
                    oldClusterIndex = clusterIndices[i2]
                    for i in xrange(numConfidentIndices):
                        if clusterIndices[i] == oldClusterIndex:
                            clusterIndices[i] = clusterIndices[i1]
                    numClusters -=1
                    
        self.detectionResult.numClusters = numClusters
    
    def release(self):
        self.windows = None
        self.numWindows = 0
