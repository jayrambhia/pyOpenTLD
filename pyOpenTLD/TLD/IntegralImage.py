import cv2
class IntegralImage:
    data = []
    width = 0
    height = 0
    
    def __init__(self, size):
        self.size = size 
        self.data = [0.0]*(size[0]*size[1])
        
    def calcIntImg(self, img, squared=False):
        output = self.data
        ip = img.getGrayNumpy()
        img = img.getMatrix()
        #for i in range(img.height):
            #for j in range(img.width):
        """
                A = cv2.cv.CreateMat(1,self.size[0],cv2.cv.CV_8UC3)
                if i >0:
                    A = output[img.height *j+i-1]
                B = cv2.cv.CreateMat(1,self.size[0],cv2.cv.CV_8UC3)
                if j>0:
                    B = output[img.height*(j-1)+i]
                C = cv2.cv.CreateMat(1,self.size[0],cv2.cv.CV_8UC3)
                if i>0 and j>0:
                    C = output[img.height*(j-1)+i-1]
                value = ip[img.step*j+i]
                if squared:
                    value = value**2
                print A
                print B
                print C
                print value
                output[img.height*j+i] = A[:]+B[:]-C[:]+value[:]
        """
        """
                if i > 0 and j > 0:
                    A = output[i-1]
                    B = output[-1+i]
                    C = output[-1+i-1]
                    value = ip[i]
                    if squared:
                        value = value**2
                    output[i] = A + B - C + value
                elif i == 0 and j > 0:
                    B = output[(j-1)+i]
                    value = ip[j+i]
                    if squared:
                        value = value**2
                    output[j+i] = B + value
                elif i > 0 and j == 0:
                    A = output[j+i-1]
                    value = ip[j+i]
                    if squared:
                        value = value**2
                    output[j+i] = A + value
                else:
                    value = ip[j+i]
                    if squared:
                        value = value**2
                    output[j+i] = value
        """
        for i in range(img.height):
            A = output[i]
            #B = output[-1+i]
            #C = output[-1+i-1]
            value = ip[i]
            if squared:
                value = value**2
            #output[i] = A + B - C + value
            output[i] = A + value
                    
        self.data = output
        
        
