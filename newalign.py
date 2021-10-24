"""New Align

Usage:
   newalign.py --version
   newalign.py [-d] rectfind <image>
   newalign.py [-d] align <outfile> [--aspect] <width> <height> [--interp=<scale>] <filename>...      
"""

import sys
#import PIL
import cv2
from docopt import docopt
import numpy as np

DEBUG = False

def dbgshow(img, win='debug'):
        cv2.imshow(win, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def rectfind(image, edgepctofbox=0.70, scalefactor = 4): #Should output ac list of corners coords in floats (UL, UR, BR, BL)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #if DEBUG:
        #dbgshow(img)
    mh, mw = img.shape[:2]
    newsize = (int(scalefactor*mw), int(scalefactor*mh))
    img = cv2.resize(img, newsize, cv2.INTER_LANCZOS4)
    ret, thresh = cv2.threshold(img,127, 255, 0)
    thresh = cv2.bitwise_not(thresh)
    if DEBUG:
        #dbgshow(thresh)
        cv2.imwrite("debug.jpg", thresh)
    im, contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imwrite("cont.jpg", cv2.dilate(im))
    bigc = None
    area = 0
    for cnt in contours:
        locx, locy, w, h = cv2.boundingRect(cnt)
        if w*h > area: 
                area = w*h 
                bigc = cnt
    
    locx, locy, w, h = cv2.boundingRect(bigc)

    #print ( locx, locy, w, h)
    #orig = cv2.imread(image, cv2.IMREAD_COLOR)
    #orig=cv2.resize(orig, newsize)
    #cv2.rectangle(orig,(locx, locy), (locx+w,locy+h), (0,255,0),)
    #cv2.drawContours(orig, [bigc], 0, (255,0,0), 1)
    

    #find edge lines of bigcnt
    xstart = locx + w*(1-edgepctofbox)/2
    xend = xstart+ w*edgepctofbox
    xmid = xstart + w/2
    ystart = locy + h*(1-edgepctofbox)/2
    yend = ystart + h*edgepctofbox
    ymid = ystart + h/2

    #print (bigc)
    toppts = []
    botpts =[]
    leftpts =[]
    rtpts =[]
    for p in bigc:
        pp = p[0]
        x,y = pp
        #print(x,y)
        addp =[x*1.0, y*1.0]
        if x >xstart and x<xend:
            if y < ymid:
                toppts.append(addp)
            else:
                botpts.append(addp)
        if y > ystart and y < yend:
            if x<xmid:
                leftpts.append(addp)
            else:
                rtpts.append(addp)
    rows, cols = thresh.shape[:2]
    toppts = np.array(toppts)
    botpts = np.array(botpts)
    rtpts = np.array(rtpts)
    leftpts = np.array(leftpts)
    #[vx,vy,x,y] =cv2.fitLine(toppts,cv2.DIST_L2, 0, .005, .005)
    lines = ( cv2.fitLine(toppts,cv2.DIST_L2, 0, .005, .005),
       cv2.fitLine(rtpts,cv2.DIST_L2, 0, .005, .005), 
       cv2.fitLine(botpts,cv2.DIST_L2, 0, .005, .005),
       cv2.fitLine(leftpts,cv2.DIST_L2, 0, .005, .005) )
    #for line in lines:
    #    drawmyline( orig, line, (255,255,255))
    #cv2.imwrite("rectangle.jpg", orig)
    output =(
        lineIntersect(lines[3], lines[0]),
        lineIntersect(lines[0], lines[1]),
        lineIntersect(lines[1], lines[2]),
        lineIntersect(lines[2], lines[3])
    )

    return(np.array(output, dtype=np.single))

def drawmyline(img, linedata, color):
    [vx,vy,x,y] = linedata
    rows, cols = img.shape[:2]
    mult = max(rows, cols)
    cv2.line(img, (x-mult*vx, y-mult*vy), (x+mult*vx, y+mult*vy), color, 1)       

def origdrawmyline(img, linedata, cols, color):
    [vx,vy,x,y] = linedata
    lefty = int((-x*vy/vx) + y)
    righty =int(((cols-x)*vy/vx)+y)
    cv2.line(img, (cols-1, righty), (0, lefty), color, 5)

def lineIntersect(line1, line2):
    t=100
    [[vx], [vy], [x1], [y1]] = line1
    x2 = x1 + t*vx
    y2 = y1 + t*vy
    [[vx], [vy], [x3], [y3]] = line2
    x4 = x3 +t*vx
    y4 = y3 +t*vy
    det = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    xret = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/det
    yret = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/det
    #return np.array([xret, yret], dtype=np.single)
    return (xret, yret)

#size is in rows, cols
def align(files, size, outputfile, scale):
    #corners = [rectfind(file, scalefactor=scale)  fogo r file in files ]
    img = np.zeros([size[0],size[1],3],dtype=np.single)
    ptsto = np.float32([[0,0], [size[1]-1,0], [size[1]-1, size[0]-1], [0, size[0]-1] ])
    for file in files:
        corner = rectfind(file, scalefactor=scale)
        print(corner)
        print(ptsto)
        M = cv2.getPerspectiveTransform( corner, ptsto )
        image = cv2.imread(file,cv2.IMREAD_COLOR)
        if DEBUG: 
            cv2.imwrite("orig.jpg", image)
        image = image.astype(np.float32)
        sh = image.shape[:2]
        image = cv2.resize(image, (scale*sh[1],scale*sh[0]), cv2.INTER_LANCZOS4)
        dst = cv2.warpPerspective(image, M, (size[0], size[1]) )   
        img += dst
    #np.multiply(img, 1.0/size(files), img)
    img= cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(outputfile, img)  

if __name__ == "__main__":
    arguments = docopt(__doc__,version='newalign .001')
    DEBUG = arguments["-d"]
    print(arguments)
    if arguments["rectfind"]:
        corners = rectfind(arguments["<image>"])
        print(corners)
    if arguments["align"]:
        scale = 4
        if arguments["--interp"] is not None:
            scale = int(arguments["--interp"])
        sz = (int(arguments["<height>"]), int(arguments["<width>"])  )
        align(arguments["<filename>"], sz, arguments["<outfile>"], scale)    



    #get image list from args, desired size of output image 
    #choose first image as base image
    #find quadrangles in image 
    #find largest quadrangle with corners
    #show corners marked on scaled image
    #let user adjust corners to match up on scaled image
    #create a template around each corner and keep
    #for each other image:
    #   load and scale image
    #   for each corner in base image:
    #       match to each new image and get coordinates of best match
    #       store for future processing
    # create result float image of requested size = 0
    #For all images:
    #   use newfound coordinates to map each image to output image and add
    #Divide result image by number of images 
    #show result and save
    




