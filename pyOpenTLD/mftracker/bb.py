from median import *
import numpy as np
def calculateBBCenter(bb):
    """
    
    **SUMMARY**
    
    Calculates the center of the given bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    center - A tuple of two floating points
    
    """
    center = (0.5*(bb[0] + bb[2]),0.5*(bb[1]+bb[3]))
    return center
    
def getFilledBBPoints(bb, numM, numN, margin):
    """
    
    **SUMMARY**
    
    Creates numM x numN points grid on Bounding Box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    numM - Number of points in height direction.
    numN - Number of points in width direction.
    margin - margin (in pixel)
    
    **RETURNS**
    
    pt - A list of points (pt[0] - x1, pt[1] - y1, pt[2] - x2, ..)
    
    """
    pointDim = 2
    bb_local = (bb[0] + margin, bb[1] + margin, bb[2] - margin, bb[3] - margin)
    if numM == 1 and numN == 1 :
        pts = calculateBBCenter(bb_local)
        return pts
    
    elif numM > 1 and numN == 1:
        divM = numM - 1
        divN = 2
        spaceM = (bb_local[3]-bb_local[1])/divM
        center = calculateBBCenter(bb_local)
        pt = [0.0]*(2*numM*numN)
        for i in range(numN):
            for j in range(numM):
                pt[i * numM * pointDim + j * pointDim + 0] = center[0]
                pt[i * numM * pointDim + j * pointDim + 1] = bb_local[1] + j * spaceM
                
        return pt
        
    elif numM == 1 and numN > 1:
        divM = 2
        divN = numN - 1
        spaceN = (bb_local[2] - bb_local[0]) / divN
        center = calculateBBCenter(bb_local)
        pt = [0.0]*((numN-1)*numM*pointDim+numN*pointDim)
        for i in range(numN):
            for j in range(numN):
                pt[i * numM * pointDim + j * pointDim + 0] = bb_local[0] + i * spaceN
                pt[i * numM * pointDim + j * pointDim + 1] = center[1]
        return pt
        
    elif numM > 1 and numN > 1:
        divM = numM - 1
        divN = numN - 1
    
    spaceN = (bb_local[2] - bb_local[0]) / divN
    spaceM = (bb_local[3] - bb_local[1]) / divM

    pt = [0.0]*((numN-1)*numM*pointDim+numM*pointDim)
    
    for i in range(numN):
        for j in range(numM):
            pt[i * numM * pointDim + j * pointDim + 0] = float(bb_local[0] + i * spaceN)
            pt[i * numM * pointDim + j * pointDim + 1] = float(bb_local[1] + j * spaceM)
    return pt

def getBBWidth(bb):
    """
    
    **SUMMARY**
    
    Get width of the bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    width of the bounding box
    
    """
    return bb[2]-bb[0]+1
    
def getBBHeight(bb):
    """
    
    **SUMMARY**
    
    Get height of the bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    height of the bounding box
    
    """
    return bb[3]-bb[1]+1
    
def predictBB(bb0, pt0, pt1, nPts):
    """
    
    **SUMMARY**
    
    Calculates the new (moved and resized) Bounding box.
    Calculation based on all relative distance changes of all points
    to every point. Then the Median of the relative Values is used.
    
    **PARAMETERS**
    
    bb0 - Bounding Box represented through 2 points (x1,y1,x2,y2)
    pt0 - Starting Points
    pt1 - Target Points
    nPts - Total number of points (eg. len(pt0))
    
    **RETURNS**
    
    bb1 - new bounding box
    shift - relative scale change of bb0
    
    """
    ofx = []
    ofy = []
    for i in range(nPts):
        ofx.append(pt1[i][0]-pt0[i][0])
        ofy.append(pt1[i][1]-pt0[i][1])
    
    dx = getMedianUnmanaged(ofx)
    dy = getMedianUnmanaged(ofy)
    ofx=ofy=0
    
    lenPdist = nPts * (nPts - 1) / 2
    dist0=[]
    for i in range(nPts):
        for j in range(i+1,nPts):
            temp0 = ((pt0[i][0] - pt0[j][0])**2 + (pt0[i][1] - pt0[j][1])**2)**0.5
            temp1 = ((pt1[i][0] - pt1[j][0])**2 + (pt1[i][1] - pt1[j][1])**2)**0.5
            dist0.append(float(temp1)/temp0)
            
    shift = getMedianUnmanaged(dist0)
    
    s0 = 0.5 * (shift - 1) * getBBWidth(bb0)
    s1 = 0.5 * (shift - 1) * getBBHeight(bb0)
    
    if shift == 0:
        shift = 1
    x1 = bb0[0] - s0 + dx
    y1 = bb0[1] - s1 + dy
    x2 = bb0[2] + s0 + dx
    y2 = bb0[3] + s1 + dy
    w = x2-x1
    h = y2-y1
    #print x1,x2,y1,y2,w,h
    if x1 <= 0 or x2 <=0 or y1<=0 or y2 <=0 or w <=20 or h <=20:
        x1 = bb0[0]
        y1 = bb0[1]
        x2 = bb0[2]
        y2 = bb0[3]
        
    bb1 = (int(x1),int(y1),int(x2),int(y2))
              
    return (bb1, shift)
    
def getBB(pt0,pt1):
    xmax = np.max((pt0[0],pt1[0]))
    xmin = np.min((pt0[0],pt1[0]))
    ymax = np.max((pt0[1],pt1[1]))
    ymin = np.min((pt0[1],pt1[1]))
    return xmin,ymin,xmax,ymax
    
def getRectFromBB(bb):
    return bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]
    
