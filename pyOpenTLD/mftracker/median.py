from copy import copy
def getMedianUnmanaged(a):
    if not a:
        return None
    newl = copy(a)
    newl.sort()
    while True:
        try:
            newl.remove(0)
        except ValueError:
            if newl:
                return newl[len(newl)/2]
            return 0

def getMedian(a):
    median = getMedianUnmanaged(a)
    return median