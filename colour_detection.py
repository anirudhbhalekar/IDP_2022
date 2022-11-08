import cv2 
import time 
import matplotlib.pyplot as plt
import os
import numpy as np 

img_path = "1_pink_arrow.jpg"

img = cv2.imread(img_path)

DIM=(1016, 760)
K=np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
D=np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])

h,w = img.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

hsv = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2HSV)

print(hsv[518, 750])

#green
#upper = np.array([31, 270, 255])
#lower = np.array([29, 90,	0])

#pink
upper = np.array([165, 180, 255])
lower = np.array([155, 100,	160])

mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(undistorted_img, undistorted_img, mask= mask)

cv2.imshow('frame', undistorted_img)
cv2.waitKey(0)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.imshow('res',res)
cv2.waitKey(0)
#cv2.imwrite("green1.jpg", mask)