import cv2
import numpy as np
from numpy.core.fromnumeric import mean
import time
backSub = cv2.createBackgroundSubtractorKNN()
capture = cv2.VideoCapture("output.mp4")
field2D = cv2.imread('2D_field.png')

p1 = (143,167)          # left corner
p2 = (1136,116)         # right corner
p3 = (639,109)          # top corner
p4 = (872,778)          # down corner

points1 = np.array([p1,p2,p3,p4], dtype=np.float32)
points2 = np.array([(165,150),(885,150),(525,0),(525,700)], dtype=np.float32)

H = cv2.getPerspectiveTransform(points1, points2)
t1 = time.time()
z = 0
while True:
    ret, frame = capture.read()
    z = z + 1
    if frame is None:
        break
    dst = cv2.GaussianBlur(frame,(1,1),cv2.BORDER_DEFAULT)
    fgMask = backSub.apply(dst)
    
    ret, fgMask = cv2.threshold(fgMask,150,255,cv2.THRESH_BINARY)
    #cv2.imshow('fgMask', fgMask)

    kernel = np.ones((3,3),np.uint8)
    T = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3,3),np.uint8)
    T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernel)

    n,C,stats, centroids = cv2.connectedComponentsWithStats(T)

    J = frame.copy()
    K = field2D.copy()
    for j, ls in enumerate(centroids):
        if str(ls[0]) == 'nan' or str(ls[1]) == 'nan':
            continue
        if stats[j][4] < 100:
            continue
        lfx = stats[j][0]
        tpx = stats[j][1]
        rtx = stats[j][0] + stats[j][2]
        dnx = stats[j][1] + stats[j][3]
        if(stats[j][2] > stats[j][3]):
            continue
        b = J[tpx:dnx, lfx:rtx,0].copy()
        r = J[tpx:dnx, lfx:rtx,2].copy()
        g = J[tpx:dnx, lfx:rtx,1].copy()
        color = [0,0,0]
        if(mean(r) > mean(b) * 1.2 and mean(r)+mean(b)>150):
            color = [0,0,255]
        else:
            color = [255,0,0]

        cv2.rectangle(J, (lfx, tpx), (rtx, dnx), color, 2)
        cv2.circle(J, ((int)(ls[0]), (int)(ls[1])), 3, color, 2)

        p = np.array([ls[0],ls[1],1])
        tp = np.matmul(H,p)
        point2Dfield = ((int)(tp[0]/tp[2]),(int)(tp[1]/tp[2]))
        
        if point2Dfield[0]>0 and point2Dfield[1]>0:
            cv2.circle(K, point2Dfield, 3, color, 2)

    cv2.imshow('Video', frame)
    cv2.imshow('Detected Video', J)
    cv2.imshow('field2D', K)
    
    keyboard = cv2.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break

t2 = time.time()
print(t1)
print(t2)
print(z)
FPS = z/(t2 - t1)
print(FPS)