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
p1,p2,p3,p4,r0,g0 = None, None, None, None, None, None
to1, to2 = None, None
block, prev_block = None, None
dec_val_list = []


initialisation_length = 20
theta = 83.5
#theta = 88
prev_angle = 90
x, y, angle, rotation, phase, f_count, g_count = 0, 0, 0, 0, 0 , -1, -1

prev_rotation, prev_distance = 0, 0
##############################################


def block_retrieval(isOn = False, thresh = 10, phase = phase):

    dec_val_list = []
    isLowDensity = False

    forward_write = "111255"
    while isOn: 

        # Writing for the robot to stop and then move forward for a bit (helps align)
        ser.write(b"0", encoding = 'utf-8') 
        ser.write(bytes(str(forward_write), encoding='utf8'))
        time.sleep(1)

        # Writing for action 4 to be executed (arduino writes US value from transmitter)
        ser.write(b"4", encoding = 'utf-8')
        raw_read = ser.read(2).decode('utf-8')
        splice_read = str(raw_read)[4:-1]

        time.sleep(0.5)
        # Reduce polling speed

        if len(splice_read > 0): 
            dec_val = int(splice_read, base=16)
            dec_val_list.append(dec_val)
        else: 
            pass
        
        # Once we have a sufficient number of distance polls in our list

        if len(dec_val_list) > 8: 
            break
    # add stuff for pincer to pinch
        
    dec_val_list.sort()
    spliced_dec_list = dec_val_list[0:3]
    
    # checks if the average of the minimum values is less than the set threshold
    
    if len(spliced_dec_list) > 0: 
        if sum(spliced_dec_list)/len(spliced_dec_list) < thresh: 
            print("LOW DENSITY BLOCK")
            isLowDensity = True
        else: 
            print("HIGH DENSITY BLOCK")
            isLowDensity = False
    else: 
        # we can tell it to re-run it here
        print("FAILED")

    return phase+1,isLowDensity
    

def rotation_and_distance_to_target(target, phase, arrow_x, arrow_y, arrow_angle, f_count, g_count, block_ready = False):

    rotation = 0
    distance = 0

    if type(target) == str:
        if target == "line_up":
            target = (arrow_x - 12, arrow_y - 200)
            try:
                distance, rotation = dt.dir_head(target[0], target[1], arrow_x, arrow_y, arrow_angle)
            except:
                pass

            if rotation > 180:
                rotation = rotation - 360
            if abs(rotation) < 1:
                phase += 1
                serial_data = bytes(str("0"), encoding='utf8')
                ser.write(serial_data)
                time.sleep(0.3)


        if target == "forwards":
            rotation = 0
            distance = 500
            print("Forwards: ", f_count)
            if f_count == 0:
                phase += 1

            #for i in range(20):
            #    cap.grab()
        if target == "grab":
            rotation = 0 
            distance = 500
            print("Grab")
            serial_data = bytes(str(310), encoding='utf8')
            ser.write(serial_data)
            if g_count == 0:
                phase += 1
    else:
    
        distance, rotation = dt.dir_head(target[0], target[1], arrow_x, arrow_y, arrow_angle)
        if distance < 30 and not block_ready:
            phase += 1
        elif distance < 40 and block_ready: 
            phase, _ = block_retrieval(True, 10, phase)

    
    if rotation > 180:
        rotation = rotation - 360

    return rotation, distance, phase

##############################################
while cap.isOpened(): 
    
    for i in range(4):
        cap.grab()
            
    ret, frame = cap.retrieve()
    
    fix_frame = dt.rotate_image(dt.undistort(frame), theta)
    h,w,_ = fix_frame.shape

    frame3 = fix_frame

    frame2 = st.filter_crop(fix_frame)
    frame2 = st.detect_edge(frame2)

    if count <= initialisation_length:
        serial_data = bytes(str("20"), encoding='utf8')
        ser.write(serial_data)

        block = dt.blue_blocks_start(fix_frame)
        print("block, : ", block)
        block = block[0]

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

        c1f = c1
        c2f = c2

        tt1 = st.stable_marker(t1, to1, count)
        tt2 = st.stable_marker(t2, to2, count)


    else: 
        c1,c2,c3,c4 = p1,p2,p3,p4
        rp, gp = r0, g0
        tt1, tt2 = to1, to2  
        block = prev_block

        c1f = (c1[0] - phase1_fudge, c1[1])
        c2f = (c2[0] - phase1_fudge, c2[1])
    

    corners, ids, rejected = cv2.aruco.detectMarkers(fix_frame, dt.arucoDict, parameters=dt.arucoParams)
    block_ready = False
    phase1_fudge = 40 

    target_list = [c1f, c2f, (block[0] - 100, block[1] + 10), (block[0], block[1] + 10), "grab", c3, (tt2[0] + 15, tt2[1] + 70), (tt2[0] + 15, tt2[1] + 40), "line_up", "forwards", c4, (rp[0], rp[1] - 50), "release"]
    target = target_list[phase]
    print(target)

    if target == "forwards" and f_count < 0:
        f_count = 15

    if target == "grab" and g_count < 0:
        g_count = 30
        serial_data = bytes(str("0"), encoding='utf8')
        ser.write(serial_data)
        time.sleep(0.1)
        serial_data = bytes(str("21"), encoding='utf8')
        ser.write(serial_data)
    
    if target == "grab":
        g_count -= 1
        print("grab ", g_count)

    if target == "release":
        serial_data = bytes(str(20), encoding='utf8')
        ser.write(serial_data)
        time.sleep(0.05)
        serial_data = bytes(str(310), encoding='utf8')
        ser.write(serial_data)
        time.sleep(0.05)
        serial_data = bytes(str(320), encoding='utf8')
        ser.write(serial_data)
        time.sleep(0.05)
        serial_data = bytes(str(0), encoding='utf8')
        ser.write(serial_data)
        ser.close()

    if target == "detect" and len(dec_val_list) < 8:
        ser.write(b"4")
        raw_read = ser.read(2).decode('utf-8')
        splice_read = str(raw_read)[4:-1]
        print("raw:", raw_read)
        print("spliceL", splice_read)
        time.sleep(0.2)
        
        if len(splice_read) > 0: 
            dec_val = int(splice_read, base=16)
            dec_val_list.append(dec_val)
        else: 
            pass

    
    if target == "detect" and len(dec_val_list) >= 8: 
        dec_val_list.sort()
        spliced_dec_list = dec_val_list[0:3]
    
        if sum(spliced_dec_list)/len(spliced_dec_list) < thresh: 
            print("LOW DENSITY BLOCK")
            density_indicator_serial = bytes(str("50"), encoding='utf-8')
            ser.write(density_indicator_serial)
        else: 
            print("HIGH DENSITY BLOCK")
            density_indicator_serial = bytes(str("51"), encoding='utf-8')
            ser.write(density_indicator_serial)
        phase += 1
        dec_val_list = []
    
    if target == block: 
        block_ready = True
    
    else: 
        block_ready = False

    print(dec_val_list)

    Cx, Cy, angle = dt.get_pose(corners, ids)

    if Cx == 0 and Cy ==0 and angle == 0:
        pass
    else:
        rotation, distance, phase = rotation_and_distance_to_target(target, phase, Cx, Cy, angle, f_count, g_count)

    f_count -= 1
    thresh = 5
    x = 0.6
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
 
    
    serial_data = bytes(str(command), encoding='utf8')
    if count > initialisation_length + 10 and (target != "grab" or target != "detect" or target != "release"):
        try: 
            x = None
            ser.write(serial_data)
        except: 
            print("Not connected!")
            pass

    #################################################################################################
    #GRAPHING
    st.plot_point(frame3,c1)
    st.plot_point(frame3,c2)
    st.plot_point(frame3,c3)
    st.plot_point(frame3,c4)
    
    st.plot_point(frame3, rp, color=(100,0,250))
    st.plot_point(frame3, gp, color=(100,0,250))
    st.plot_point(frame3, tt1,color= (0,0,250))
    st.plot_point(frame3, tt2,color= (0,0,250))
    try:
        st.plot_point(frame3, target, color = (0, 0, 0))
        st.plot_point(frame3, block, (255,255,0))
        cv2.arrowedLine(frame3, (Cx,Cy), target, (0,255,0), 2)
    except:
        pass
    
    
    try: 
        frame3 = st.plot_hline(frame3, int(h/2))
        frame3 = st.plot_vline(frame3, int(w/2)) 
        frame3 = st.plot_hline(frame3, int(h/4))
    except: 
        pass 

    #END GRAPHING 
    ####################################################################################################
    
    p1,p2,p3,p4 = c1,c2,c3,c4
    r0,g0 = rp, gp
    to1, to2 = tt1, tt2
    prev_block = block

    count += 1
    phase = phase % len(target_list)
    prev_target = target

    try: 
        frame3 = dt.aruco_display(fix_frame, corners, ids, rejected=None)
        cv2.imshow('stream', frame3)
    except: 
        cv2.imshow('stream', fix_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            serial_data = bytes("0", encoding='utf8')
            try: 
                ser.write(serial_data)
            except: 
                pass
            break

cap.release()
cv2.destroyAllWindows()