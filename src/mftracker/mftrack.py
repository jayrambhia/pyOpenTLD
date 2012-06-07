from SimpleCV import Camera, Image, Display
from fbtrack import *
from bb import getBB, getRectFromBB

def mftrack():
    cam = Camera()
    p1 = None
    p2 = None
    d = Display()
    while d.isNotDone():
        try:
            img = cam.getImage()
            a=img.save(d)
            dwn = d.leftButtonDownPosition()
            up = d.leftButtonUpPosition()
            
            if dwn:
                p1 = dwn
            if up:
                p2 = up
                break

            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    if not p1 or not p2:
        return None
    
    bb = getBB(p1,p2)
    rect = getRectFromBB(bb)
    img.drawRectangle(rect[0],rect[1],rect[2],rect[3],width=5)
    i = img.copy()
    img.save(d)
    time.sleep(0.5)
    img1 = cam.getImage()
    while True:
        try:
            newbb, shift = fbtrack(i.getGrayNumpy(),img1.getGrayNumpy(), bb, 12, 12, 3, 12)
            print newbb, shift
            rect = getRectFromBB(bb)
            img1.drawRectangle(rect[0],rect[1],rect[2],rect[3],width=5)
            img1.save(d)
            time.sleep(0.1)
            i = img1.copy()
            bb = newbb
            img1 = cam.getImage()
        except KeyboardInterrupt:
            break
