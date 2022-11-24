import stream_test as st
import detection_tools as dt
import numpy as np
import cv2
import time
import serial
##############################################

ser = serial.Serial("COM9", 9600)

ser.close()    
ser.open()
timeout = ser.timeout
ser.timeout = 2

##############################################

url = "http://localhost:8081/stream/video.mjpeg"
count = 1 
cap = cv2.VideoCapture(url)

initialisation_length = 20
theta = 83.5
#theta = 88
prev_angle = 90
phase1_fudge = 40
x, y, angle, rotation, phase, f_count, g_count = 0, 0, 0, 0, 0 , -1, -1
prev_rotation, prev_distance = 0, 0
##############################################
stable = dt.initialise(cap, theta, True)
rp, gp, c1, c2, c3, c4, tt1, tt2 = stable

c1f = (c1[0] - phase1_fudge, c1[1])
c2f = (c2[0] - phase1_fudge, c2[1])
block = (0, 0)
print(stable)
while cap.isOpened(): 
#for i in range(100):
    Cx, Cy, angle, fix_frame, block = dt.vision(cap, theta, phase, block)

    target_list = [c1f, c2f, (block[0] - 100, block[1] + 10), (block[0], block[1] + 10), "grab", c3, 
    (tt2[0] + 15, tt2[1] + 70), (tt2[0] + 15, tt2[1] + 40), "line_up", "forwards", c4, (rp[0], rp[1] - 50), "release"]
    target = target_list[phase]
    
    #print(target)
    command, distance, rotation = dt.get_command(target, Cx, Cy, angle)
    serial_data = bytes(str(command), encoding='utf8')
    
    #ser.write(serial_data)

    update, count = dt.update_handler(target, distance, rotation, count)
    phase += update
    fix_frame = dt.GUI(fix_frame, stable, target, block, Cx, Cy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        serial_data = bytes("0", encoding='utf8')
        try: 
            ser.write(serial_data)
        except: 
            pass
        break

    cv2.imshow('stream', fix_frame)