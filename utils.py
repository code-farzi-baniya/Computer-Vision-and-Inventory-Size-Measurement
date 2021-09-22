import cv2 as cv 
import numpy as np


# Function to get the contours from our image :
def getContours(img,cannyThr = [100,100],showCanny =False,minArea = 1000,filter = 0,draw = False):
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv.Canny(imgBlur,cannyThr[0],cannyThr[1])
    
    kernel = np.ones((7,7))
    imgDial = cv.dilate(imgCanny,kernel,iterations=3)
    # eroding process after dialation
    imgThreshold = cv.erode(imgDial,kernel,iterations=2)
    
    if showCanny:
        cv.imshow("Canny Image",imgThreshold)

    contours, hierarchy = cv.findContours(imgThreshold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    finalContours = []
    for i in contours:
        area = cv.contourArea(i)
        if area > minArea: # minarea is user defined
            peri = cv.arcLength(i,True)
            approx = cv.approxPolyDP(i,0.02*peri,True)
            boundingbox = cv.boundingRect(approx)
            
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx),area,approx,boundingbox,i])
            else:
                finalContours.append([len(approx),area,approx,boundingbox,i])
    
    finalContours = sorted(finalContours,key = lambda x:x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv.drawContours(img,con[4],-1,(0,0,255),4)
    
    return img, finalContours


def reorder(myPoints):
    print(myPoints.shape)
    myPointsnew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsnew[0] = myPoints[np.argmin(add)]
    myPointsnew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis = 1)
    myPointsnew[1] = myPoints[np.argmin(diff)]
    myPointsnew[2] = myPoints[np.argmax(diff)]
    return myPointsnew
    

def warpImage(img,points,w,h,pad=10):
    # print(points)
    # print(reorder(points))
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    
    return imgWarp


def findDist(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 +(pts2[1]-pts1[1])**2)**0.5