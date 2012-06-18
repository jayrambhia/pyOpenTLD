from SimpleCV import Image, Camera, Display
from pyOpenTLD import *

class PyOpenTLD:
    tld = TLD()
    display = Display()
    cam = Camera()
    threshold = 0
    initialBB = []
    
    def __init__(self):
        self.threshold = 0.5
        self.initialBB = []
        
    def start_tld(self, bb=None):
        #self.tld = TLD()
        img = self.cam.getImage()
        grey = img.toGray().getBitmap()
        
        self.tld.detectorCascade.imgWidth = grey.width
        self.tld.detectorCascade.imgHeight = grey.height
        self.tld.detectorCascade.imgWidthStep = grey.width*grey.nChannels
        if not bb:
            bb = getBBFromUser(self.cam, self.display)
        print bb
        grey = img.toGray()
        self.tld.selectObject(grey, bb)
        skipProcessingOnce = True
        reuseFrameOnce = True
        
        self.tld.processImage(img)
        if self.tld.currBB:
            print self.tld.currBB
            x,y,w,h = self.tld.currBB
            img.drawRectangle(x,y,w,h,width=5)
            img.show()
            #return self.tld.currBB
        self.start_tld(self.tld.currBB)
            #time.sleep(2)

def getBBFromUser(cam, d):
    p1 = None
    p2 = None
    while d.isNotDone():
        try:
            img = cam.getImage()
            a=img.save(d)
            dwn = d.leftButtonDownPosition()
            up = d.leftButtonUpPosition()
            
            if dwn:
                p1 = dwn
            if up:
                p2 = up
                break

            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    if not p1 or not p2:
        return None
    
    bb = getBB(p1,p2)
    rect = getRectFromBB(bb)
    return rect

p=PyOpenTLD()
p.start_tld()
        
    
