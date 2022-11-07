import cv2 
import time 
import matplotlib.pyplot as plt
import os

import numpy as np 

url = "http://localhost:8081/stream/video.mjpeg"

filename = "orientatio"

try: 
    if not os.path.exists(filename):
        os.makedirs(filename)
except OSError: 
    print("Error - not created")

count = 0 
print("OK")
while count < 20: 
    a = input("trigger")
    if a == 'h':
        cap = cv2.VideoCapture(url)
        ret,frame = cap.read()
        cv2.imwrite("./{}/{}_pink_arrow".format(filename,count) + ".jpg", frame)   
        cap.release() 
        count += 1 

cv2.destroyAllWindows()