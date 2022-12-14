import cv2 
import time 
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import glob
import numpy as np 

url = "http://localhost:8081/stream/video.mjpeg"

theta2 = 0
#theta2 = 4 
theta = 83.5 + theta2

# Camera parameters for distortion and the camera matrix + dimensions
DIM=(1016, 760)
K= np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
D=np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])

def rotate_image(image, angle):
    
    # rotates the image by a value of theta
    image_center = tuple(np.array(image.shape[1::-1]) / 2)  # computes center of image
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0) # derives rotation matrix about center       
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR) # rotates 
    return result

def undistort(img, balance=0.0, dim2=None, dim3=None):

    # to remove fish eye distortion (values we got from tests)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1

    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
   
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img

# edge detection
def detect_edge(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# mask creation for white path lines
def mask(frame):
    img =  (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL))
    sensitivity = 40    
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])    

    mask = cv2.inRange(img, lower_white, upper_white)
    return mask

def region_of_interest(edges, top_height_parameter = 0.2, bot_height_parameter = 0.1):
    
    # this function will mask regions of interest by ANDing a polygon of that shape

    # percent to shave off the top
    # percent to shave off the bottom

   
    h,w = edges.shape
    mask = np.zeros_like(edges) 
    
    # Polygon creation

    polygon = np.array([[
        (0, h * top_height_parameter),
        (w, h * top_height_parameter),
        (w, h * (1 - bot_height_parameter)),
        (0, h * (1 - bot_height_parameter)),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    return cropped_edges

# creation of houghlines (based on edges detected from white path lines)
def houghline(frame, minL):
    rho = 1 # Pixel precision value
    angle = np.pi/180 # radian precision value 
    min_votes = 10 # min number of votes hough lines needs

    line_segments = cv2.HoughLinesP(frame, rho, angle, min_votes, None, minLineLength= minL, maxLineGap=3)

    
    return line_segments

# plot lanes (houghlines) - this is for testing and visualisation purposes
def plot_lanes(superimpose, lines, color = (250,150,0), l_width = 2, thresh = 1.01):
    corner_line_arr = []
    zone_line_arr = []

    # We initialise a corner and zone array to store corner and zone points
    h,w = superimpose.shape[0], superimpose.shape[1]

    if lines is not None: 
        for line_seg in lines: 
            for x1, y1, x2, y2 in line_seg:                                
                line_color = color
                # ignores divide by zero errors
                if max(x2,x1)/min(x1,x2) > thresh: 
                    grad = (y2 - y1)/(x2 - x1)

                    # this means the abs value of the gradient is about 1 (corner)
                    if abs(grad) < 1.5 and abs(grad) > 0.5: 
                        line_color = (0,0,255) # changes the color of that marker
                        corner_line_arr.append(line_seg) # adds it to the corner array
                    
                if max(y2,y1)/min(y2,y1) > thresh:
                    grad = (x2 - x1)/(y2 - y1)
                    # ignores grad being infty 
                    # if the gradient is near vertical (or the inverse is about 0) and the marker is in the upper 25% of the image
                    if abs(grad) < 0.1 and min(y2,y1) < h/4: 
                        line_color = (100,50,100)
                        zone_line_arr.append(line_seg) # adds it to the zone array
                    

                
                line_image = cv2.line(superimpose, (x1, y1), (x2, y2), line_color, l_width)  
    else: 
        print("NONE DETECTED")
        line_image = np.zeros_like(superimpose)

    #line_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)  
    return line_image, corner_line_arr, zone_line_arr

# filtering 
def filter_crop(image): 
    img = region_of_interest(mask(image))
    return img

# color detection for any arbitrary color
def detect_colour(img, upper, lower, ret_colour = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if not ret_colour:
        return mask
    else:
        return np.expand_dims(mask, -1) * img

# object detection from contour image (applies to specific colors)
def detect_obj(contours):
    data = np.array(contours[:, 0, :], dtype = np.float64)
    initial = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data, initial)
    angle = 360 / (2 * np.pi) * np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    return mean, angle

def detect_objects(binary_img, min_area = 0, max_area = 999999):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mean_list = []
    angle_list = []

    for contour in contours:      
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            mean, angle = detect_obj(contour)
            mean_list.append(mean)
            angle_list.append(angle)
    return mean_list, angle_list

# plots a point
def plot_point(image, coords, color = (0,255,0)): 
    l = 5
    if coords is not None: 
        bot = (coords[0] - l, coords[1] - l)
        top = (coords[0] + l, coords[1] + l)
        image = cv2.rectangle(image, bot, top,color,-1)
 
    return image

# plots a rectangle
def plot_rectangle(img, l = 10, coords = (100,100)):

    h,w = img.shape[1], img.shape[0]
    bot = (coords[0] - l, coords[1] - l)
    top = (coords[0] + l, coords[1] + l)
    img = cv2.rectangle(img, bot, top,(0,255,0),1)

    return img

# plots a vertical line
def plot_vline(img, x): 
    h,w = img.shape[0], img.shape[1]
    img = cv2.line(img, (x,0),(x,h), color=(0,0,255), thickness=2)

    return img

# plots a horizontal line
def plot_hline(img, y): 
    h,w = img.shape[0], img.shape[1]    
    img = cv2.line(img, (0,y),(w,y), color=(0,0,255), thickness=2)

    return img

# finds the corner markers in an ordered fashion
def find_markers(img,lines):
    h,w = img.shape[0], img.shape[1]
    m1,m2,m3,m4 = None, None, None, None
    if lines is not None:
        for line in lines: 
            # for each quadrant of the image (it iterates over the corner array found earlier)
            for x1,y1,x2,y2 in line:
                if x1 <= int(w/2) and y1 <= int(h/2):
                    m1 = (int((x1+x2)/2),int((y1+y2)/2))
                elif x1 <= int(w/2) and y1 > int(h/2):
                    m2 = (int((x1+x2)/2),int((y1+y2)/2))
                elif x1 > int(w/2) and y1 > int(h/2):
                    m3 = (int((x1+x2)/2),int((y1+y2)/2))
                elif x1 > int(w/2) and y1 <= int(h/2):
                    m4 = (int((x1+x2)/2),int((y1+y2)/2))
                else: continue
        
    return m1,m2,m3,m4

# finds the red and green zones
def find_zones(img, zone_array):
    
    h,w = img.shape[0], img.shape[1]
    g,r = None, None
    min = int(h)
    if zone_array is not None: 
        for line in zone_array: 
            for x1,y1,x2,y2 in line: 
                
                if np.minimum(y1,y2) <= min: 
                    min = np.minimum(y1,y2)
                
                if x1 > int(w/2) + 100: 
                    g = (int((x1+x2)/2), min) 
                elif x1 < int(w/2) - 100: 
                    r = (int((x1+x2)/2), min) 
                else: 
                    pass 
            
    return g,r 

# finds the tunnel markers 
# (the exit of the tunnel is the lowest vertical line point in the upper right quadrant)
# (the entrance of the tunnel is the highest vertical line point in the lower right quadrant) 
def tunnel_marker(img, all_lines): 

    h,w = img.shape[0], img.shape[1]
    
    tunnel_top, tunnel_bot = None, None
    max_y, min_y = (0,0), (0, int(h)) 
    z_max, z_min = 0, int(h)

    if all_lines is not None: 
        for line in all_lines: 
            for x0,y0,x1,y1 in line: 

                x_m, y_m = int((x0 + x1)/2) , int((y0 + y1)/2)

                if x_m > w/2 and y_m < h/2: 
                    if y_m > z_max: 
                        max_y = (x_m, np.maximum(y0,y1))
                        z_max = y_m

                if x_m > w/2 and y_m > h/2: 
                    if y_m < z_min: 
                        min_y = (x_m, np.minimum(y0,y1))
                        z_min = y_m
        
    return max_y, min_y

# Marker stabilisation algorithm 
# takes the current and previous values and returns a value that is an interpolation between the two 
# it will divide the interpolation by the count so as the count increases the marker 'stabilizes'
# requires that there is a sequence of uninterrupter footage for the markers to get a solid position and estimation of ground truth

def stable_marker(curr_markers, prev_markers, count):
    
    if curr_markers is None and prev_markers is not None: 
        return prev_markers
    elif curr_markers is None and prev_markers is None: 
        return None
    elif curr_markers is not None and prev_markers is None: 
        return curr_markers
    else:
        x0,y0 = prev_markers[0], prev_markers[1]
        x1,y1 = curr_markers[0], curr_markers[1]

        if count > 500:
            return prev_markers
        else: 
            assert count != 0 
            xp = (x1 - x0) * np.sqrt(1)/np.sqrt(np.sqrt(count*10))
            yp = (y1 - y0) * np.sqrt(1)/np.sqrt(np.sqrt(count*10))
            x_diff = int(xp)
            y_diff = int(yp)

            xn  = x0 + x_diff
            yn = y0 + y_diff

            return (xn,yn)

# simple distance function
def distance(coord1,coord2):
    d = -1
    if coord1 and coord2 is not None: 
        x1,y1 = coord1[0], coord1[1]
        x2,y2 = coord2[0], coord2[1]

        x_d = abs(x2 - x1)
        y_d = abs(y2 - y1)
        d = np.sqrt(np.square(y_d) + np.square(x_d))

    return d

