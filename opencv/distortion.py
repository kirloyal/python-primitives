import numpy as np
import cv2

w, h = 1200, 800
data = np.full((h,w,3), 200, dtype=np.uint8)

cx0, cy0 = 200, 200
cw, ch =  50, 50
cgrid = 7,10

for i in range(cgrid[0]):
    for j in range(cgrid[1]):
        pt1 = (cx0+j*cw, cy0+i*ch)
        pt2 = (cx0+(j+1)*cw, cy0+(i+1)*ch)
        c = ((i+j)%2 * 255, (i+j)%2 * 255, (i+j)%2 * 255)
        cv2.rectangle(data, pt1, pt2, c, -1)

cv2.imshow('img', data)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints = [] 
imgpoints = [] 

gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY) 
gray
ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

print(ret, corners)
if ret == True:
    objpoints.append(objp)

    corners2=cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    cv2.drawChessboardCorners(data, (9,6), corners2, ret)
    cv2.imshow('img', data)
    cv2.waitKey(0)

cv2.destroyAllWindows()
