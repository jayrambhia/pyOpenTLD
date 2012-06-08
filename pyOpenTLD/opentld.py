from SimpleCV import Image, Camera, Display

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
        
    
