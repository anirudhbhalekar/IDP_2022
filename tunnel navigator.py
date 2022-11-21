import stream_test as st
import detection_tools as dt
import numpy as np
import cv2
import time
import serial

ser = serial.Serial("COM9", 9600)

ser.close()
ser.open()

url = "http://localhost:8081/stream/video.mjpeg"
count = 1 
cap = cv2.VideoCapture(url)
p1,p2,p3,p4,r0,g0 = None, None, None, None, None, None
to1, to2 = None, None

initialisation_length = 20
theta = 83.5
#theta = 88
prev_angle = 90
phase = 0
rotation = 0

while cap.isOpened(): 
    
    for i in range(4):
        cap.grab()
            
    ret, frame = cap.retrieve()
    
    fix_frame = dt.rotate_image(dt.undistort(frame), theta)
    h,w = fix_frame.shape[0], fix_frame.shape[1]
    frame3 = fix_frame

    frame2 = st.filter_crop(fix_frame)
    frame2 = st.detect_edge(frame2)

    if count <= initialisation_length:
        
        line_segments = st.houghline(frame2, 10)
        frame3, corners, zones = st.plot_lanes(fix_frame,line_segments)
    
        m1,m2,m3,m4 = st.find_markers(frame3, corners)
        r1,g1 = st.find_zones(frame3, zones)
        t1,t2 = st.tunnel_marker(frame3, line_segments)


        rp = st.stable_marker(r1, r0, count)
        gp = st.stable_marker(g1, g0, count)
        
        c1 = st.stable_marker(m1,p1,count)
        c2 = st.stable_marker(m2,p2,count)
        c3 = st.stable_marker(m3,p3,count)
        c4 = st.stable_marker(m4,p4,count)

        tt1 = st.stable_marker(t1, to1, count)
        tt2 = st.stable_marker(t2, to2, count)

    else: 
        c1,c2,c3,c4 = p1,p2,p3,p4
        rp, gp = r0, g0
        tt1, tt2 = to1, to2  

    try:
        x, y, angle, prev_angle  = dt.corrected_pink_arrow(fix_frame, prev_angle)
        #print(x, y)
    except IndexError:
        a = 1
        #print("Arrow not detected")

    target_list = [(tt2[0], tt2[1] + 100), "line_up", "forwards", c1]
    target = target_list[phase]


    if type(target) == str:
        if target == "line_up":

            target = tt1
            command = "0"
            serial_data = bytes(str(command), encoding='utf8')
            ser.write(serial_data)
            
            #frame3, distance, rotation, prev_angle = dt.plot_pink_arrow_direction(frame3, target[0], target[1], prev_angle)
            #if rotation > 180:
            #    rotation = rotation - 360
            #if rotation < 3:
            #    phase += 1

        if target == "forwards":
            command = "111255"
            serial_data = bytes(str(command), encoding='utf8')
            ser.write(serial_data)
            time.sleep(2)
            phase += 1



    else:
        try:
            frame3, distance, rotation, prev_angle = dt.plot_pink_arrow_direction(frame3, target[0], target[1], prev_angle)
            if distance < 30:
                phase += 1
        except:
            a = 1
    
    if rotation > 180:
        rotation = rotation - 360

    thresh = 5
    x = 0.7
    speed = int(abs(rotation) * 255/ 180 * x + 255 * (1 - x))
    speed = f"{speed:03d}"

    if rotation < -1 * thresh:
        #turn left
        command = "101" + speed
    elif rotation > thresh:
        #turn right
        command = "110" + speed
    else:
        command = "111255"
        #command = "0"

    print(command)
    serial_data = bytes(str(command), encoding='utf8')
    if count > initialisation_length + 10:
        ser.write(serial_data)

    st.plot_point(frame3,c1)
    st.plot_point(frame3,c2)
    st.plot_point(frame3,c3)
    st.plot_point(frame3,c4)
    
    st.plot_point(frame3, rp, color=(100,0,250))
    st.plot_point(frame3, gp, color=(100,0,250))
    st.plot_point(frame3, tt1,color= (0,0,250))
    st.plot_point(frame3, tt2,color= (0,0,250))
    st.plot_point(frame3, target, color = (0, 0, 0))

    frame3 = st.plot_hline(frame3, int(h/2))
    frame3 = st.plot_vline(frame3, int(w/2)) 
    frame3 = st.plot_hline(frame3, int(h/4))

    #st.plot_rectangle(frame3)
    
    p1,p2,p3,p4 = c1,c2,c3,c4
    r0,g0 = rp, gp
    to1, to2 = tt1, tt2

    count += 1
    phase = phase % len(target_list)
    prev_target = target

    cv2.imshow('stream', frame3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            serial_data = bytes("0", encoding='utf8')
            ser.write(serial_data)
            break

cap.release()
cv2.destroyAllWindows()
    
    