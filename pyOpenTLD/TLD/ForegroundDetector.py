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
        print img
        img.show()
        #time.sleep(1)
        #absImg = img
        #threshImg = cv2.threshold(absImg,self.fgThreshold,255,cv2.THRESH_BINARY)
        print self.fgThreshold
        threshImg = img.threshold(self.fgThreshold)
        threshImg.show()
        #//blobs = CBlobResult(im, None, 0)
        blobs = threshImg.findBlobs()
        #//blobs.Filter( blobs, B_EXCLUDE, CBlobGetArea(), B_LESS, minBlobSize )
        fgList = self.detectionResult.fgList
        
        for blob in blobs:
            fgList.append(blob.getBoundingBox())
            
        self.detectionResult.fgList = fgList
        
    def isActive(self):
        return (not self.bgImg)
        
    def release(self):
        pass
