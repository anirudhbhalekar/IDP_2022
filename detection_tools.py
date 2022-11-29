import numpy as np
import cv2
import vis_tools as vt
import math
import time
import stream_test as st
import serial

# parameters for camera 1
# DIM specifies the dimensions, K and D are the camera and distortion matrices respectively
DIM = (1016, 760)
K = np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
D = np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])


# ARUCO dictionary (stores the IDs and patterns for all aruco markers) - is references when we initialise which ArUco marker type
#  - the robot has 
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

aruco_type = "DICT_4X4_250"

# Using a 4x4 ArUco marker with ID 138 

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()

# Displays ArUCo GUI
def aruco_display(image, corners, ids, rejected): 
    
    if len(corners) > 0: 
        ids = ids.flatten()

        # Iterates over all - unpacked - corners and ids detected

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4,2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Converts the above into an ordered tuple
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Plots appropriate lines to form a bounding box
            cv2.line(image, topLeft, topRight, (0,255,0), 3)
            cv2.line(image, topRight, bottomRight, (0,255,0), 3)
            cv2.line(image, bottomRight, bottomLeft, (0,255,0), 3)
            cv2.line(image, bottomLeft, topLeft, (0,255,0), 3)


            # Gets the mid points of the top and bottom lines (for vector plotting)
            bottomMid = (int((topRight[0] + topLeft[0])/2.0), int((topRight[1] + topLeft[1])/2.0))
            topMid = (int((bottomLeft[0] + bottomRight[0])/2.0), int((bottomLeft[1] + bottomRight[1])/2.0))

            # Gets the center coordinates for the marker

            cX = int((topLeft[0] + bottomRight[0])/2.0)
            cY = int((topLeft[1] + bottomRight[1])/2.0)
            cv2.circle(image, (cX,cY), 8, (0,0,255), -1)

            # Plots an arrowed line in the direction of the heading
            cv2.arrowedLine(image, topMid, bottomMid, (255,0,0), 2)
            angle = return_angle(topMid, bottomMid)

            # Text for the angle and ID detected
            cv2.putText(image, str(angle), (topMid[0], topMid[1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

            return image

# Method to detect and return all corners and ids of detected ArUco markers (using openCV inbuilt)
def aruco_detection(img): 

    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    return corners, ids

# Returns angle between two coordinates (takes arctangent)
def return_angle(coords0, coords1): 
    x0,y0 = coords0[0], coords0[1]
    x1,y1 = coords1[0], coords1[1]

    try:
        angle = (180/np.pi) * np.arctan((y1-y0)/(x1-x0))
    except:
        # Divide by zero exception
        angle = 90
    
    # Some orientation manipulation to make it easier to debug 
    angle = 90 - abs(angle)

    if x0 < x1 and y0 < y1:
        angle = 360 - angle

    if x0 < x1 and y0 >= y1:
        angle = 180 + angle

    if x0 >= x1 and y0 >= y1:
        angle = 180 - angle

    return angle

# Method to return center coordinates and angle of ArUco markers (in a seperate method to make debugging easier)
def get_pose(corners, ids): 

    cX, cY, angle = 0,0,0

    if len(corners) > 0: 
        ids = ids.flatten()
    
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4,2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            topMid = (int((topRight[0] + topLeft[0])/2.0), int((topRight[1] + topLeft[1])/2.0))
            bottomMid = (int((bottomLeft[0] + bottomRight[0])/2.0), int((bottomLeft[1] + bottomRight[1])/2.0))

            cX = int((topLeft[0] + bottomRight[0])/2.0)
            cY = int((topLeft[1] + bottomRight[1])/2.0)

            angle = return_angle(topMid, bottomMid)

    return cX, cY, angle
    
def undistort(img):
    #function to remove fisheye from an image taken with the overhead camera
    dim1 = img.shape[:2][::-1]  

    scaled_K = K * dim1[0] / DIM[0]
    scaled_K[2][2] = 1.0  

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim1, np.eye(3), balance = 0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img

def detect_colour(img, upper, lower, ret_colour = False):
    #returns a map of pixels with colour values which lie between the upper and lower bounds
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if not ret_colour:
        return mask
    else:
        return np.expand_dims(mask, -1) * img

def binary_image(img, cutoff = 50):
    #returns a binary map of the inputed image with a greyscale cutoff as specified
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a, binary = cv2.threshold(grey, cutoff, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def detect_obj(contours):
    #detects a single object from a contour map and returns the pixel values of the coordinates
    #of the centroid and heading 
    data = np.array(contours[:, 0, :], dtype = np.float64)
    initial = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data, initial)
    angle = 360 / (2 * np.pi) * np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    return mean, angle

def detect_objects(binary_img, min_area = 0, max_area = 999999):
    #detects all objects with area inbetween the min and max values and returns the pixel values of the coordinates
    #of the centroid and heading for each

    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mean_list = []
    angle_list = []

    for contour in contours:      
        area = cv2.contourArea(contour)
        #print(area)
        if area >= min_area and area <= max_area:
            mean, angle = detect_obj(contour)
            mean_list.append(mean)
            angle_list.append(angle)
    return mean_list, angle_list

def rotate_image(image, angle):
    #returns inputed image rotated by the specified angle
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)       
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# Finds the blue blocks based on color detection - works well with blue colors 

def find_blue_blocks(undistorted_img, return_blue = False):
    upper = np.array([120, 360, 360])
    lower = np.array([100, 0,	0])

    # Upper lower HSV thresholds for blue

    blues = detect_colour(undistorted_img, upper, lower, False)

    # Blue isolated mask
    centre_list, _ = detect_objects(blues, 70, 100000)

    # Finds weighted mean centre coordinates of mask (if blue detected, otherwise it returns the previous value)
    if return_blue:
        return centre_list, blues
    else:
        return centre_list

def no_filter_crop(image): 
    img = region_of_interest(detect_edge(mask(image)))
    return img

def region_of_interest(edges):
    
    # this function will mask regions of interest by ANDing a polygon of that shape

    top_height_parameter = 0.2 # percent to shave off the top
    bot_height_parameter = 0.1  # percent to shave off the bottom

    h,w = edges.shape
    mask = np.zeros_like(edges) 
    
    # Polygon creation

    polygon = np.array([[
        (0, h * top_height_parameter),
        (w, h * top_height_parameter),
        (w, h * (1 - bot_height_parameter)),
        (0, h * (1 - bot_height_parameter)),
    ]], np.int32)

    # bitwise ANDing
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    return cropped_edges

# Canny edge detection
def detect_edge(image):
    edges = cv2.Canny(image, 200, 400)
    return edges

# Mask generation (for white path line detection)
def mask(frame):
    img =  (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL))
    sensitivity = 15
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])    

    mask = cv2.inRange(img, lower_white, upper_white)
    return mask

# returns the distance to and relative angle between the target and the robot pose

def dir_head(target_x, target_y, arrow_x, arrow_y, arrow_angle):
    distance, rotation = 500,0
    target = (target_x, target_y)
    arrow = np.array([arrow_x, arrow_y])

    if target is not None and arrow is not None: 
        #print(target, " ", arrow)
        delta = target - arrow
        # linear distance function
        distance = np.sqrt(np.sum(np.square(delta)))
        # arctangent
        angle = math.atan(delta[1] / delta[0]) * 180 / math.pi

        if delta[0] < 0:
            angle += 180

        rotation = 720 + angle - arrow_angle
        rotation = rotation % 360

    rotation += 90
    return distance, rotation   

# Returns a sorted array of centers for the blue blocks (the ones closest to the left wall are in the beginning)
def blue_blocks_start(img):
    x = np.array([310, 725])
    y = np.array([525, 690])
    centres = find_blue_blocks(img)
    true_centres = []
    for centre in centres:
        centre = centre[0]
        if (x[0] < centre[0] < x[1]) and (y[0] < centre[1] < y[1]):
            true_centres.append((int(centre[0]), int(centre[1])))
    return sorted(true_centres, key = lambda x: x[0])

# Simple average distance finding function (high density blocks are not detectable by the US sensor)
def detect_block(dec_val_list, thresh): 
    avg_dist = sum(dec_val_list)/len(dec_val_list)
    isLowDensity = False

    if avg_dist < thresh: 
        # This means the block is low density (detectable)
        isLowDensity = True
    else: 
        isLowDensity = False

    return isLowDensity

# Ultrasound reading poll command
def ultrasound_read(ser): 
    ser.write(b"4")
    raw_read = ser.read(2)
    return raw_read

# INITIALIZATION PERIOD FOR MARKER IDENTIFICATION
##################################################

def initialise(cap, theta, show = False):
    p1,p2,p3,p4,r0,g0 = None, None, None, None, None, None
    to1, to2 = None, None
    for j in range(20):
        count = j
        for i in range(4):
            cap.grab()
                
        ret, frame = cap.retrieve()
        
        fix_frame = rotate_image(undistort(frame), theta)
        h,w,_ = fix_frame.shape

        frame2 = st.filter_crop(fix_frame)
        frame2 = st.detect_edge(frame2)

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

        p1,p2,p3,p4 = c1,c2,c3,c4 
        r0, g0 = rp, gp
        to1, to2 = tt1, tt2

        if show:
            st.plot_point(frame3,c1)
            st.plot_point(frame3,c2)
            st.plot_point(frame3,c3)
            st.plot_point(frame3,c4)
        
            st.plot_point(frame3, rp, color=(100,0,250))
            st.plot_point(frame3, gp, color=(100,0,250))
            st.plot_point(frame3, tt1,color= (0,0,250))
            st.plot_point(frame3, tt2,color= (0,0,250))

            cv2.imshow('stream', frame3)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    return rp, gp, c1, c2, c3, c4, tt1, tt2
#########################################################################

# Returns pose centers, orientation, and position of current and last blocks
def vision(cap, theta, phase, block):
    for i in range(6):
        cap.grab()
            
    ret, frame = cap.retrieve()
    
    fix_frame = rotate_image(undistort(frame), theta)
    h,w,_ = fix_frame.shape

    corners, ids, _ = cv2.aruco.detectMarkers(fix_frame, arucoDict, parameters = arucoParams)
    
    if phase == 0 or block == (0, 0):
        try:
            block = blue_blocks_start(fix_frame)[0]
            last_block = blue_blocks_start(fix_frame)[-1]
        except IndexError:
            pass

    Cx, Cy, angle = get_pose(corners, ids)
    return Cx, Cy, angle, fix_frame, block, last_block

# For non-coordinate targets - we wish to execute a routine instead - the following dictates the serial commands for each
def string_target(target, Cx, Cy):
    command = None
    if target == "forward":
        command = "111255"
    if target == "reverse":
        command = "100255"
    elif target == "grab":
        command = "21"
    elif target == "release":
        command = "20"
    elif target == "line_up":
        target = (Cx - 12, Cy - 200)
    elif target == "detect":
        command = "4"
    return target, command

# Defines the commands for the robot in relation to the target

def get_command(target, Cx, Cy, angle, prev_rotation, nudge_count, thresh = 5, x = 0.7):
    command = None
    distance, rotation = 0, 0
    if type(target) is str:
        target, command = string_target(target, Cx, Cy)

    if command is None:   
        if Cx == 0 and Cy == 0 and angle == 0:
            # These are base initial conditions (means the robot hasnt been initialised yet)
            pass
        else:
            distance, rotation = dir_head(target[0], target[1], Cx, Cy, angle)    
            if rotation > 180:
                rotation = rotation - 360
            
            # If the robot is very close to the target angle - damping will mean it essentially slows to a creep
            # we add a count factor to nudge the robot out
            if abs(rotation - prev_rotation) <= 0.1: 
                nudge_count += 1
    
            else: 
                nudge_count = 0

            speed = int(abs(rotation) * 255/ 180 * x + 255 * (1 - x) + 10*nudge_count)

            # To prevent wrap around error - if the speed is 256 - then the arduino reads 0
            speed = min(255, speed)
            speed = f"{speed:03d}"

                
            if rotation < -1 * thresh:
                command = "101" + speed
            elif rotation > thresh:
                command = "110" + speed
            else:
                command = "111255"
                nudge_count = 0

    return command, distance, rotation, nudge_count

def string_update(target, distance, rotation, count):
    update = 0
    if target == "grab":
        count -= 40
    elif target == "release":
        count -= 5
    elif target == "forward":
        count -= 5
    elif target == "reverse":
        count -= 5
    elif target == "detect":
        count -= 10
    elif target == "line_up":
        if abs(rotation) < 0.1:
            update = 1
            print(rotation)

    if count <= 0:
        update = 1
    
    return update, count


# Method to notify the program when the robot has reached the target
def update_handler(target, distance, rotation, count):
    update = 0
    if type(target) is str:
        update, count = string_update(target, distance, rotation, count)
    else:
        if distance < 30:
            update = 1
    if update == 1:
        count = 100
    return update, count

# GUI elements for ease of debugging and visual analysis
def GUI(fix_frame, stable, target, block, Cx, Cy):
    rp, gp, c1, c2, c3, c4, tt1, tt2 = stable
    corners, ids, _ = cv2.aruco.detectMarkers(fix_frame, arucoDict, parameters = arucoParams)

    st.plot_point(fix_frame,c1)
    st.plot_point(fix_frame,c2)
    st.plot_point(fix_frame,c3)
    st.plot_point(fix_frame,c4)
    
    st.plot_point(fix_frame, rp, color=(100,0,250))
    st.plot_point(fix_frame, gp, color=(100,0,250))
    st.plot_point(fix_frame, tt1,color= (0,0,250))
    st.plot_point(fix_frame, tt2,color= (0,0,250))

    try:
        st.plot_point(fix_frame, target, color = (0, 0, 0))
        st.plot_point(fix_frame, block, (255,255,0))
        cv2.arrowedLine(fix_frame, (Cx,Cy), target, (0,255,0), 2)
    except:
        pass

    fix_frame = aruco_display(fix_frame, corners, ids, rejected=None)
    return fix_frame
    