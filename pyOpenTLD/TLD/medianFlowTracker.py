#namespace TLD
from math import floor
class MedianFlowTracker:
    trackerBB = []
    
    def __init__(self):
        pass
    
    def track(self, prevImg, currImg, prevBB):
        # prevBB -> x1,y1,w,h
        # bb_tracker -> x1,y1,x2,y2
        if prevBB:
            if prevBB[2] <= 0 or prevBB[3] <= 0 :
                return
            bb_tracker = [prevBB[0], prevBB[1], prevBB[0]+prevBB[2]-1, prevBB[1]+prevBB[3]-1]
            
            bb_tracker, shift = fbtrack(prevImg, currImg, bb_tracker)
            
            x = floor(bb_tracker[0]+0.5)
            y = floor(bb_tracker[1]+0.5)
            w = floor(bb_tracker[2]-bb_tracker[0]+1+0.5)
            h = floor(bb_tracker[3]-bb_tracker[1]+1+0.5)
            
            if x<0 or y<0 or w<=0 or h<=0 or x +w > currImg.cols or y+h > currImg.rows or not x or not y or not w or not h:
                pass
            else:
                self.trackerBB = [x,y,w,h]
            
            

        
