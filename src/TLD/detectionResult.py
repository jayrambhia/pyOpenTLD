#namespace tld
class detectionResult:
    containsValidData = False
	fgList = []
	confidentIndices = []
	numClusters = 0
	detectorBB = None
	variances = None
	posteriors = None
	featureVectors = None
    
    def __init__(self, numWindows, numTrees):
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
        
