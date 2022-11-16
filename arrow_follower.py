import cv2 
import time 
import matplotlib.pyplot as plt
import os
import numpy as np 

import detection_tools as dt
import vis_tools as vt

url = "http://localhost:8081/stream/video.mjpeg"

cap = cv2.VideoCapture(url)
theta = 83.2
DIM=(1016, 760)
K= np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
D=np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])

arrow_x = []
arrow_y = []
arrow_angle = []

prev_angle = 90


while cap.isOpened(): 
    for i in range(4):
        cap.grab()

    ret, frame = cap.retrieve()
    undistorted_img = dt.undistort(frame)
    rotated = dt.rotate_image(undistorted_img, theta)
    edge = dt.edge_detection(rotated)

    #"""
    try:
    #if True:
        arrowed = dt.arrow_to_all_blocks(rotated, prev_angle)
        #print(distance, rotation, prev_angle)
        #centres, arrowed = dt.find_blue_blocks(rotated, True)
        #print(centres)
    except IndexError:
        arrowed = rotated 
        
    """     

    try:
        _, arrowed = dt.find_blue_blocks(rotated, True)
        cv2.imshow('stream', arrowed)
    except:
        a = 1

    """    
    #a = input("Hello")

    cv2.imshow('stream', arrowed)
    #cap.release()
    #cap = cv2.VideoCapture(url)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

plt.plot(arrow_x, arrow_y)
plt.show()
plt.plot(arrow_angle)
plt.show()

c1, c2, c3, c4, gp, rp = (245, 209), (339, 616), (769, 614), (789, 200), (708, 151), (315, 151)
