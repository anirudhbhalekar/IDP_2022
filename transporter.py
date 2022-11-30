import stream_test as st
import detection_tools as dt
import numpy as np
import cv2
import time
import keyboard
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

before_home = (int((rp[0] + gp[0])/2), gp[1])
home = (int((rp[0] + gp[0])/2), gp[1] - 90)
c1f = (c1[0] - phase1_fudge, c1[1])
c2f = (c2[0] - phase1_fudge, c2[1])

final_target = c1f
penultimate_target = c1f

ramp_m = (int((c1f[0] + c2f[0])/2) - 10, int((c1f[1] + c2f[1])/2))
block, last_block = (0, 0), (0,0)
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
    Cx, Cy, angle, fix_frame, block, last_block = dt.vision(cap, theta, phase, block, last_block)

    avoid_target = (last_block[0], last_block[1] - 75)
    #target_list = ["detect"]
    target_list = [c1f, ramp_m, c2f, (block[0] - 100, block[1] + 11), (block[0], block[1] + 11), "grab", "detect", avoid_target, c3, 
    (tt2[0] + 15, tt2[1] + 70), (tt2[0] + 12, tt2[1] + 40), "line_up", "line_up", "forward", (c4[0] + 25, c4[1]), c4, xp, (xp[0], xp[1] - 50), "release", "reverse", 
    penultimate_target, final_target]
    #target_list =  ["detect"]
    target = target_list[phase]

    if block == last_block and target == avoid_target:
        phase += 1
        target = target_list[phase]

    command, distance, rotation, nudge_counter = dt.get_command(target, Cx, Cy, angle, prev_rotation, nudge_counter)
    prev_rotation = rotation
    update, count = dt.update_handler(target, distance, rotation, count)

    if update == 1:
        count = 100
        command = "0"

        if target == "detect":
            try:
                isLowDensity = dt.detect_block(dist_list, 35)
                print(isLowDensity)
            except ZeroDivisionError:
                isLowDensity = False
            if isLowDensity:
                xp = rp
                command = "311"
            else:
                xp = gp
                command = "301"

            for i in range(10):
                cap.grab()

        if target == "release":
            if isLowDensity:
                xp = rp
                command = "310"
            else:
                xp = gp
                command = "300"
        
        dist_list = []
            

    serial_data = bytes(str(command), encoding='utf8')
    ser.write(serial_data)


    if type(target) == None: 
        phase += 1

    if target == "detect":
        raw_read = ser.read(4)
        print(raw_read)
        splice_read = str(raw_read)[4:-1]
        if len(splice_read) > 0:
            try: 
                dec_val = int(splice_read, base=16)
                print(dec_val) 
                if dec_val is not None: 
                    dist_list.append(dec_val)
                    print(dec_val)
                else: 
                    dist_list.append(255)
            except: 
                count += 10


        if len(dist_list) >= 2:
            count = -1
    
    if target == home and update == 1:
        ser.write(bytes("0", encoding='utf8'))
        break 

    phase += update
    phase = phase % len(target_list)

    frame_3 = dt.GUI(fix_frame, stable, target, block, Cx, Cy)

    try:
        cv2.imshow('stream', frame_3)
    except:
        cv2.imshow('stream', fix_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        ser.write(bytes("0", encoding='utf8'))
        break

    if keyboard.is_pressed('h'):
        final_target = home
        penultimate_target = before_home
        print("pressed")
        print("GOING HOME")

print("FINISHED")
