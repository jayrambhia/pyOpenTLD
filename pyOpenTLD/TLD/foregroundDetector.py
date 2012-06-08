import cv2
from pyOpenTLD.TLD.detectionResult import *
#import BlobResult
class ForegroundDetector:
    
    fgThreshold = 16
    minBlobSize = 0
    bgImg = None
    detectionResult = DetectionResult()
    
    def __init__(self, fgThreshold=16, minBlobSize=0):
        self.fgThreshold = fgThreshold
        self.minBlobSize = minBlobSize
        self.bgImg = bgImg
    
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
