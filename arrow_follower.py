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
prev_true_angle = prev_angle + 180
angle_tolerance = 70
while cap.isOpened(): 
    for i in range(4):
        cap.grab()

    ret, frame = cap.retrieve()
    undistorted_img = dt.undistort(frame)
    rotated = dt.rotate_image(undistorted_img, theta)
    edge = dt.edge_detection(rotated)

    search_region = dt.no_filter_crop(rotated)
    rotated_corners, _ = dt.detect_corners(search_region, rotated)
    try:
        x, y, angle = dt.find_pink_arrow(rotated, edge)
        delta_angle = angle - prev_angle
        dir_delta = delta_angle / abs(delta_angle)
        
        if abs(delta_angle) > 90:
            print(angle, prev_angle)
            angle = prev_angle + (abs(delta_angle) - 180) * dir_delta
            print(angle)
        prev_angle = angle
        #angle = true_angle - 180
        arrow_x.append(x)
        arrow_y.append(y)
        arrow_angle.append(angle)

        arrowed = vt.draw_arrow(x, y, angle, rotated_corners)

    except IndexError:
        arrowed = rotated      

    #a = input("Hello")

    cv2.imshow('stream', arrowed)
    #cap.release()
    #cap = cv2.VideoCapture(url)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

print(arrow_x)
print(arrow_y)
print(len(arrow_x))
print(arrow_angle)

cap.release()
cv2.destroyAllWindows()

plt.plot(arrow_x, arrow_y)
plt.show()
plt.plot(arrow_angle)
plt.show()

c1, c2, c3, c4, gp, rp = (245, 209), (339, 616), (769, 614), (789, 200), (708, 151), (315, 151)
