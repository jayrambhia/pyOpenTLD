import cv2
import cv2.cv as cv
import math
import time
import numpy as np
import scipy.spatial.distance as spsd

def lktrack(img1, img2, ptsI, nPtsI, winsize_ncc=10, win_size_lk=4, method=cv2.cv.CV_TM_CCOEFF_NORMED):
    """
    **SUMMARY**
    
    Lucas-Kanede Tracker with pyramids
    
    **PARAMETERS**
    
    img1 - Previous image or image containing the known bounding box (Numpy array)
    img2 - Current image
    ptsI - Points to track from the first image
           Format ptsI[0] - x1, ptsI[1] - y1, ptsI[2] - x2, ..
    nPtsI - Number of points to track from the first image
    winsize_ncc - size of the search window at each pyramid level in LK tracker (in int)
    method - Paramete specifying the comparison method for normalized cross correlation 
             (see http://opencv.itseez.com/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#cv2.matchTemplate)
    
    **RETURNS**
    
    fb - forward-backward confidence value. (corresponds to euclidean distance between).
    ncc - normCrossCorrelation values
    status - Indicates positive tracks. 1 = PosTrack 0 = NegTrack
    ptsJ - Calculated Points of second image
    
    """
    template_pt = []
    target_pt = []
    fb_pt = []
    ptsJ = [-1]*len(ptsI)
    
    for i in range(nPtsI):
        template_pt.append((ptsI[2*i],ptsI[2*i+1]))
        target_pt.append((ptsI[2*i],ptsI[2*i+1]))
        fb_pt.append((ptsI[2*i],ptsI[2*i+1]))
    
    template_pt = np.asarray(template_pt,dtype="float32")
    target_pt = np.asarray(target_pt,dtype="float32")
    fb_pt = np.asarray(fb_pt,dtype="float32")
    
    target_pt, status, track_error = cv2.calcOpticalFlowPyrLK(img1, img2, template_pt, target_pt, 
                                     winSize=(win_size_lk, win_size_lk), flags = cv2.OPTFLOW_USE_INITIAL_FLOW,
                                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                                     
    fb_pt, status_bt, track_error_bt = cv2.calcOpticalFlowPyrLK(img2,img1, target_pt,fb_pt, 
                                       winSize = (win_size_lk,win_size_lk),flags = cv2.OPTFLOW_USE_INITIAL_FLOW,
                                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    status = status & status_bt
    ncc = normCrossCorrelation(img1, img2, template_pt, target_pt, status, winsize_ncc, method)
    fb = euclideanDistance(template_pt, target_pt)
    
    newfb = -1*np.ones(len(fb))
    newncc = -1*np.ones(len(ncc))
    for i in np.argwhere(status):
        i = i[0]
        ptsJ[2 * i] = target_pt[i][0]
        ptsJ[2 * i + 1] = target_pt[i][1]
        newfb[i] = fb[i]
        newncc[i] = ncc[i]

    return newfb, newncc, status, ptsJ
    
def euclideanDistance(point1,point2):
    """
    **SUMMARY**
    
    Calculates eculidean distance between two points
    
    **PARAMETERS**
    
    point1 - vector of points
    point2 - vector of points with same length
    
    **RETURNS**
    
    match = returns a vector of eculidean distance
    """
    match = ((point1[:,0]-point2[:,0])**2+(point1[:,1]-point2[:,1])**2)**0.5
    return match

def normCrossCorrelation(img1, img2, pt0, pt1, status, winsize, method=cv2.cv.CV_TM_CCOEFF_NORMED):
    """
    **SUMMARY**
    
    Calculates normalized cross correlation for every point.
    
    **PARAMETERS**
    
    img1 - Image 1.
    img2 - Image 2.
    pt0 - vector of points of img1
    pt1 - vector of points of img2
    status - Switch which point pairs should be calculated.
             if status[i] == 1 => match[i] is calculated.
             else match[i] = 0.0
    winsize- Size of quadratic area around the point
             which is compared.
    method - Specifies the way how image regions are compared. see cv2.matchTemplate
    
    **RETURNS**
    
    match - Output: Array will contain ncc values.
            0.0 if not calculated.
 
    """
    nPts = len(pt0)
    match = np.zeros(nPts)
    for i in np.argwhere(status):
        i = i[0]
        patch1 = cv2.getRectSubPix(img1,(winsize,winsize),tuple(pt0[i]))
        patch2 = cv2.getRectSubPix(img2,(winsize,winsize),tuple(pt1[i]))
        match[i] = cv2.matchTemplate(patch1,patch2,method)
    return match
