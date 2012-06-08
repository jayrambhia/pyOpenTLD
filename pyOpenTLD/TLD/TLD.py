import cv2
from random import shuffle
#namspace tld
class TLD:
    trackEnabled = False
    detectorEnabled = False
    learningEnabled = False
    alternating = False
    
    medianFlowTracker = MedianFlowTracker()
    detectorCascade = DetectorCascade()
    nnClassifier = NNClassifier()
    
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
                            self.detectionResult.featureVectors[detectorCascade.numTrees*idx:])
        
        for i in range(numIterations):
            idx = positiveIndices[i][0]
            self.detectorCascade.ensembleClassifier.learn(self.currImg,
                            self.detectorCascade.windows[TLD_WINDOW_SIZE*idx:],
                            True,
                            self.detectionResult.featureVectors[detectorCascade->numTrees*idx:])
                            
        for i in range(len(negativeIndicesForNN)):
            idx = negativeIndiceForNN[i]
            patch = NormalizedPatch()
            patch.values = tldExtractNormalizedPatchBB(currImg, self.detectorCascade.windows[TLD_WINDOW_SIZE*idx:]
            patch.positive = 0
            patches.append(patch)
            
        self.detectorCascade.nnClassifier.learn(patches)
            
                    
            
                
            
            
        
            
        
    
