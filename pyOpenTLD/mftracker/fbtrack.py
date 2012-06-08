from lk import *
from bb import *
from median import *

def fbtrack(imgI, imgJ, bb, numM=10, numN=10,margin=5,winsize_ncc=10):
    """
    **SUMMARY**
    
    Forward-Backward tracking using Lucas-Kanade Tracker
    
    **PARAMETERS**
    
    imgI - Image contain Object with known BoundingBox (Numpy array)
    imgJ - Following image (Numpy array)
    bb - Bounding box represented through 2 points (x1,y1,x2,y2)
    numM - Number of points in height direction.
    numN - Number of points in width direction.
    margin - margin (in pixel)
    winsize_ncc - size of the search window at each pyramid level in LK tracker (in int)
    
    **RETURNS**
    
    newbb - Bounding box of object in track in imgJ
    scaleshift - relative scale change of bb
    
    """
    nPoints = numM*numN
    sizePointsArray = nPoints*2
    
    pt = getFilledBBPoints(bb, numM, numN, margin)
    fb, ncc, status, ptTracked = lktrack(imgI, imgJ, pt, nPoints, winsize_ncc)
    
    nlkPoints = 0
    for i in range(nPoints):
        nlkPoints += status[i][0]

    startPoints = []
    targetPoints = []
    fbLKCleaned = [0.0]*nlkPoints
    nccLKCleaned = [0.0]*nlkPoints
    M = 2
    nRealPoints = 0
    
    for i in range(nPoints):
        if ptTracked[M*i] is not None:
            startPoints.append((pt[2 * i],pt[2*i+1]))
            targetPoints.append((ptTracked[2 * i], ptTracked[2 * i + 1]))
            fbLKCleaned[nRealPoints]=fb[i]
            nccLKCleaned[nRealPoints]=ncc[i][0][0]
            nRealPoints+=1
            
    medFb = getMedian(fbLKCleaned)
    medNcc = getMedian(nccLKCleaned)
    
    nAfterFbUsage = 0
    for i in range(nlkPoints):
        if fbLKCleaned[i] <= medFb and nccLKCleaned[i] >= medNcc:
            startPoints[nAfterFbUsage] = startPoints[i]
            targetPoints[nAfterFbUsage] = targetPoints[i]
            nAfterFbUsage+=1

    newBB, scaleshift = predictBB(bb, startPoints, targetPoints, nAfterFbUsage)
    
    return (newBB, scaleshift)

