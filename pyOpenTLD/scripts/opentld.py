from SimpleCV import Image, Camera, Display
from pyOpenTLD import *
'''
from pyOpenTLD.mftracker.bb import *
from pyOpenTLD.mftracker.lk import *
from pyOpenTLD.mftracker.fbtrack import *
from pyOpenTLD.mftracker.median import *
from pyOpenTLD.TLD.clustering import *
from pyOpenTLD.TLD.detectorCascade import *
from pyOpenTLD.TLD.ensembleClassifier import *
from pyOpenTLD.TLD.foregroundDetector import *
from pyOpenTLD.TLD.medianFlowTracker import *
from pyOpenTLD.TLD.NNClassifier import *
from pyOpenTLD.TLD.TLD import *
from pyOpenTLD.TLD.TLDUtil import *
from pyOpenTLD.TLD.varianceFilter import *
'''
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

PyOpenTLD()
        
    
