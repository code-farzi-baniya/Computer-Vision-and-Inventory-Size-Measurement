import cv2 as cv 
import numpy as np 
from utils import *

# initially
webcam = False
# Path will depend on your system : 
img_path = "/Users/mrnobody/Desktop/WORK/Projects/Fifth SEM Projects/Fom Project/img3.jpg"

cap = cv.VideoCapture(0) # Webcam object
cap.set(10,1000) # for setting the brightness
cap.set(3,1920) # for the width
cap.set(4,1080) # for setting up the height

scale = 3
wP = 210*scale
hP = 297*scale
while True:
    if webcam:
        success,image = cap.read()
    else:
        image = cv.imread(img_path)
    # parameters : pixel changes,none, scale between 0 and 1 
    image, finalContours = getContours(image,minArea=50000,filter=4)
    
    if len(finalContours)!=0:
        biggest = finalContours[0][2]
        # print(biggest," Value of Approx Length ",len_approx)
        imgWarp = warpImage(image,biggest,wP,hP)
        cv.imshow("A4 Paper",imgWarp)
        
        image2, finalContours2 = getContours(imgWarp,minArea=2000,filter=4,cannyThr=[50,50],draw = True)
        if len(finalContours)!=0:
            for obj in finalContours2:
                cv.polylines(image2,[obj[2]],True,(0,255,00),2)
                nPoints = reorder(obj[2])
                nW = round((findDist(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((findDist(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
                cv.arrowedLine(image2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv.arrowedLine(image2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv.putText(image2, '{}cm'.format(nW), (x + 30, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv.putText(image2, '{}cm'.format(nH), (x - 70, y + h // 2), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        cv.imshow("A4",image2) 
        # print(" Value of Approx Length ",len_approx)
    image = cv.resize(image,(0,0),None,0.5,0.5) # resize
    cv.imshow("Original Output",image)
    # cv.waitKey(1)
    if cv.waitKey(1) & 0xff == ord('q'): 
        break

