#%%

import os
import numpy as np

if not os.path.exists('tmp'):
    os.makedirs('tmp')

w, h = 120, 80
data = np.zeros((120,h,w,3), np.uint8)
for i in range(120):
    data[i,:,i,int(i/40)%3] = 255

#%%

import cv2

filename = 'tmp/tmp.mp4'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 25

capSize = w, h
out = cv2.VideoWriter(filename, fourcc, fps, capSize)
for frame in data:
    out.write(frame)

out.release()

#%%

import cv2

filename = 'tmp/tmp.mp4'
cap = cv2.VideoCapture(filename)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES,nFrame/2)

print(fps)
print(w,h)
print(nFrame)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
           break
    else:
        break

print("done")
cap.release()
cv2.destroyAllWindows()


#%%

