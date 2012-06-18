import cv2
import cv2.cv as cv
from SimpleCV import Image
import numpy as np
import time
class ForegroundDetector:
    
    fgThreshold = 128
    minBlobSize = 0
    bgImg = None
    
    from DetectionResult import DetectionResult
    detectionResult = DetectionResult()
    
    def __init__(self, fgThreshold=128, minBlobSize=0):
        self.fgThreshold = fgThreshold
        self.minBlobSize = minBlobSize
    
    def nextIteration(self, img):
        if not self.bgImg:
            return
        threshImg = img.threshold(self.fgThreshold)
        blobs = threshImg.findBlobs()
        fgList = self.detectionResult.fgList
        
        for blob in blobs:
            fgList.append(blob.getBoundingBox())
            
        self.detectionResult.fgList = fgList
        
    def isActive(self):
        return (self.bgImg)
        
    def release(self):
        pass
