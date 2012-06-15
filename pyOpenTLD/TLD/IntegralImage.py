class IntegralImage:
    data = []
    width = 0
    height = 0
    
    def __init__(self, size):
        self.data = [0.0]*(size[0]*size[1])
        
    def calcIntImg(self, img, squared=False):
        output = self.data
        ip = img.data
        for i in range(img.height):
            for j in range(img.width):
                A = 0
                if i >0:
                    A = output[img.height *j+i-1]
                B = 0
                if j>0:
                    B = output[img.height*(j-1)+i]
                C = 0
                if i>0 and j>0:
                    C = output[img.height*(j-1)+i-1]
                value = ip[img.step*j+i]
                if squared:
                    value = value**2
                output[img.height*j+i] = A+B-C+value
        self.data = output
        
        
