TLD_WINDOW_OFFSET_SIZE = 10
class VarianceFilter:
    from DetectionResult import DetectionResult
    detectionResult = DetectionResult()
    integralImage = []
    integralImage_squared = []
    enabled = False
    windowOffsets = []
    minVar = 0.0
    
    def __init__(self):
        self.enabled = True
        self.minVar = 0
        self.integralImage = None
        self.integralImage_squared = None
        
    def calcVariance(self, off):
        ii1 = self.integralImage.data
        ii2 = self.integralImage_squared.data
        
        mX  = (ii1[off[3]] - ii1[off[2]] - ii1[off[1]] + ii1[off[0]]) / float(off[5])
        mX2 = (ii2[off[3]] - ii2[off[2]] - ii2[off[1]] + ii2[off[0]]) / float(off[5])
        return mX2 - mX*mX;
        
    def nextIteration(self, img):
        if not self.enabled:
            return
        from IntegralImage import IntegralImage
        self.integralImage = IntegralImage(img.size())
        self.integralImage.calcIntImg(img)
        
        self.integralImage_squared = IntegralImage(img.size())
        self.integralImage_squared.calcIntImg(img, True)
        
    def filter(self, i):
        if not self.enabled:
            return True

        bboxvar = self.calcVariance(self.windowOffsets[TLD_WINDOW_OFFSET_SIZE*i:])
        self.detectionResult.variances[i] = bboxvar;

        if bboxvar < self.minVar:
            return False

        return True
        
    def release(self):
        self.integralImage = None
        self.integralImage_squared = None
