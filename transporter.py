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
phase1_fudge = 30
x, y, angle, rotation, phase, count = 0, 0, 0, 0, 0, 100
prev_rotation, prev_distance, nudge_counter = 0, 0, 0
dist_list = []
##############################################
#stable = dt.initialise(cap, theta, True)
stable = ((709, 152), (317, 152), (247, 201), (251, 610), (779, 619), (790, 205), (801, 328), (795, 503))
rp, gp, c1, c2, c3, c4, tt1, tt2 = stable

c1f = (c1[0] - phase1_fudge, c1[1])
c2f = (c2[0] - phase1_fudge, c2[1])
block = (0, 0)
xp = (0, 0)

serial_data = bytes("300", encoding='utf8')
ser.write(serial_data)
time.sleep(0.01)
serial_data = bytes("310", encoding='utf8')
ser.write(serial_data)
time.sleep(0.01)
serial_data = bytes("320", encoding='utf8')
ser.write(serial_data)
time.sleep(0.01)
serial_data = bytes("20", encoding='utf8')
ser.write(serial_data)


while cap.isOpened(): 
    Cx, Cy, angle, fix_frame, block = dt.vision(cap, theta, phase, block)

    target_list = [c1f, c2f, (block[0] - 100, block[1] + 11), (block[0], block[1] + 11), "grab", "detect", c3, 
    (tt2[0] + 15, tt2[1] + 70), (tt2[0] + 15, tt2[1] + 40), "line_up", "line_up", "forward", (c4[0] + 25, c4[1]), c4, xp, (xp[0], xp[1] - 50), "release", "reverse"]
    #target_list =  ["grab", "detect", xp, (xp[0], xp[1] - 50), "release", "reverse", c1]
    target = target_list[phase]

    command, distance, rotation, nudge_counter = dt.get_command(target, Cx, Cy, angle, prev_rotation, nudge_counter)
    update, count = dt.update_handler(target, distance, rotation, count)

    if update == 1:
        count = 100
        command = "0"

        if target == "detect":
            try:
                isLowDensity = dt.detect_block(dist_list, 15)
            except ZeroDivisionError:
                isLowDensity = False
            print(isLowDensity)
            print(dist_list)
            if isLowDensity:
                xp = rp
                command = "311"
            else:
                xp = gp
                command = "321"

            for i in range(100):
                cap.grab()

        if target == "release":
            if isLowDensity:
                xp = rp
                command = "310"
            else:
                xp = gp
                command = "320"
        
        dist_list = []
            

    serial_data = bytes(str(command), encoding='utf8')
    ser.write(serial_data)

    if target == "detect":
        raw_read = ser.read(2)
        splice_read = str(raw_read)[4:-1]
        #print(raw_read)
        if len(splice_read) > 0:
            try: 
                dec_val = int(splice_read, base=16)
                dist_list.append(dec_val)
                print(dec_val)
            except: 
                count += 10
    
    phase += update
    phase = phase % len(target_list)

    frame_3 = dt.GUI(fix_frame, stable, target, block, Cx, Cy)

    try:
        cv2.imshow('stream', frame_3)
    except:
        cv2.imshow('stream', fix_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        serial_data = bytes("0", encoding='utf8')
        try: 
            ser.write(serial_data)
        except: 
            pass
        break

    