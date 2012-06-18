import cv2
class IntegralImage:
    data = []
    width = 0
    height = 0
    
    def __init__(self, size):
        self.size = size 
        self.data = [0.0]*(size[0]*size[1]*3)
        
    def calcIntImg(self, img, squared=False):
        ip = img.getNumpy().flat
        op = self.data
        matrix = img.getMatrix()
        cols = matrix.cols
        rows = matrix.rows
        step = img.getMatrix().step
        #print step*rows+cols
        #print len(ip)
        for i in xrange(cols):
            for j in xrange(rows):
                A = 0
                B = 0
                C = 0
                if i > 0:
                    A = op[cols * j + i - 1]
                if j > 0:
                    B = op[cols * (j - 1) + i]
                if i > 0 and j > 0:
                    C = op[cols * (j - 1) + i - 1]
                value = ip[step * j + i -1]
                if squared:
                    value = value**2
                op[cols * j + i] = A + B - C + value
        self.data = op
        
        
