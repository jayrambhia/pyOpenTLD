from math import floor
TLD_WINDOW_SIZE = 5;
TLD_WINDOW_OFFSET_SIZE = 6
#namespace tld
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
        """
        self.objWidth = -1
        self.objHeight = -1
        self.useShift = 1
        self.imgHeight = -1
        self.imgWidth = -1

        self.shift=0.1
        self.minScale=-10
        self.maxScale=10
        self.minSize = 25
        self.imgWidthStep = -1

        self.numTrees = 13
        self.numFeatures = 10

        self.initialised = False
        """
        pass
        """
        self.initWindowsAndScales()
        self.initWindowOffsets()

        self.propagateMembers()

        self.ensembleClassifier->init();

        self.initialised = True
        """
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
                    windowIndex++
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
