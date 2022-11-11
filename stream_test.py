import cv2 
import time 
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

import numpy as np 

url = "http://localhost:8081/stream/video.mjpeg"

cap = cv2.VideoCapture(url)

# theta2 = 4 
theta2 = 0
theta = 84 + theta2

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

def filter(image):
    return region_of_interest(detect_edge(mask(image)))

def detect_edge(image):
    edges = cv2.Canny(image, 200, 400)
    return edges

def mask(frame):
    img =  (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL))
    sensitivity = 15
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])    

    mask = cv2.inRange(img, lower_white, upper_white)
    return mask


def region_of_interest(edges):
    
    # this function will mask regions of interest by ANDing a polygon of that shape

    top_height_parameter = 0.2 # percent to shave off the top
    bot_height_parameter = 0.1  # percent to shave off the bottom

    right_width_parameter = 0.5 # percent to shave from the right
    left_width_parameter = 0.1
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

def detect_corners(frame, superimpose): 
        
    corners= cv2.goodFeaturesToTrack(frame, 30, 0.01, 35)
    corner_arr = []
    l = 5
    if corners is not None: 
        for corner in corners:
            x,y= corner[0]
            x= int(x)
            y= int(y)
            corner_arr.append((x,y))
            superimpose = cv2.rectangle(superimpose, (x-l,y-l),(x+l,y+l),(255,0,0),-1)
    

    dst = superimpose 
    return dst, corner_arr

def detect_path(frame):
    rho = 1 # Pixel precision value
    angle = np.pi/180 # radian precision value 
    min_votes = 4 # min number of votes hough lines needs

    line_segments = cv2.HoughLinesP(frame, rho, angle, min_votes, None, minLineLength= 300, maxLineGap=3)

    
    return line_segments

def plot_lanes(frame, lines, color = (50,0,50), l_width = 2, thresh = 1.001):
    if lines is not None: 
        for line_seg in lines: 
            for x1, y1, x2, y2 in line_seg:                
                
                line_color = color
                if x2/x1 >= thresh: 
                    grad = (y2 - y1)/(x2 - x1)
                    if abs(grad) < 1.25 and abs(grad) > 0.75: 
                        line_color = (255,255,255)
                        

                line_image = cv2.line(frame, (x1, y1), (x2, y2), line_color, l_width)  
    else: 
        print("NONE DETECTED")
        line_image = np.zeros_like(frame)

    #line_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)  
    return line_image

def no_filter_crop(image): 
    img = region_of_interest(detect_edge(mask(image)))
    return img

def find_pink_arrow(undistorted_img, edge):
    upper = np.array([165, 255, 255])
    lower = np.array([155, 50,	160])
    single_colour = detect_colour(undistorted_img, upper, lower) * edge
    centre, angle = detect_objects(single_colour, 100, 10000)
    return centre[0][0][0], centre[0][0][1], angle[0]

def detect_colour(img, upper, lower, ret_colour = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if not ret_colour:
        return mask
    else:
        return np.expand_dims(mask, -1) * img

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

while cap.isOpened(): 
    
    for i in range(4):
        cap.grab()
    
    ret, frame = cap.retrieve()
    
    fix_frame = rotate_image(undistort(frame), theta)

    frame2 = no_filter_crop(fix_frame)
    frame2, corner_array = detect_corners(frame2, fix_frame)

    cv2.imshow('stream', frame2)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
