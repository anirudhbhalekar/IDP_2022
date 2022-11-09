import cv2 
import time 
import matplotlib.pyplot as plt
import os
import numpy as np 

import detection_tools as dt

url = "http://localhost:8081/stream/video.mjpeg"

cap = cv2.VideoCapture(url)
theta = 84
DIM=(1016, 760)
K= np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
D=np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])

arrow_x = []
arrow_y = []
arrow_angle = []

while cap.isOpened(): 
    for i in range(2):
        cap.grab()

    ret, frame = cap.retrieve()
    undistorted_img = dt.undistort(frame)
    rotated = dt.rotate_image(undistorted_img, 83.3 + 180)
    
    edge = dt.edge_detection(rotated)
    x, y, angle = dt.find_pink_arrow(rotated, edge)
    
    arrow_x.append(x)
    arrow_y.append(y)
    arrow_angle.append(angle)

    #a = input("Hello")

    cv2.imshow('stream', rotated)
    #cap.release()
    #cap = cv2.VideoCapture(url)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

print(arrow_x)
print(arrow_y)
print(len(arrow_x))

cap.release()
cv2.destroyAllWindows()

plt.plot(arrow_x, arrow_y)
plt.show()
plt.plot(arrow_angle)
plt.show()