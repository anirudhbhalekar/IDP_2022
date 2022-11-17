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

def detect_edge(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def mask(frame):
    img =  (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL))
    sensitivity = 40    
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])    

    mask = cv2.inRange(img, lower_white, upper_white)
    return mask

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

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    return cropped_edges

def detect_corners(frame, superimpose): 

    corners = np.zeros_like(frame)
    if frame is not None:   
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

def houghline(frame, minL):
    rho = 1 # Pixel precision value
    angle = np.pi/180 # radian precision value 
    min_votes = 10 # min number of votes hough lines needs

    line_segments = cv2.HoughLinesP(frame, rho, angle, min_votes, None, minLineLength= minL, maxLineGap=3)

    
    return line_segments

def plot_lanes(superimpose, lines, color = (250,150,0), l_width = 2, thresh = 1.01):
    corner_line_arr = []
    zone_line_arr = []

    h,w = superimpose.shape[0], superimpose.shape[1]

    if lines is not None: 
        for line_seg in lines: 
            for x1, y1, x2, y2 in line_seg:                                
                line_color = color
                if max(x2,x1)/min(x1,x2) > thresh: 
                    grad = (y2 - y1)/(x2 - x1)
                    if abs(grad) < 1.5 and abs(grad) > 0.5: 
                        line_color = (0,0,255)
                        corner_line_arr.append(line_seg)
                    
                if max(y2,y1)/min(y2,y1) > thresh:
                    grad = (x2 - x1)/(y2 - y1)
                    if abs(grad) < 0.1 and min(y2,y1) < h/4: 
                        line_color = (100,50,100)
                        zone_line_arr.append(line_seg)
                    

                
                line_image = cv2.line(superimpose, (x1, y1), (x2, y2), line_color, l_width)  
    else: 
        print("NONE DETECTED")
        line_image = np.zeros_like(superimpose)

    #line_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)  
    return line_image, corner_line_arr, zone_line_arr

def filter_crop(image): 
    img = region_of_interest(mask(image))
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

def plot_point(image, coords, color = (0,255,0)): 
    l = 5
    if coords is not None: 
        bot = (coords[0] - l, coords[1] - l)
        top = (coords[0] + l, coords[1] + l)
        image = cv2.rectangle(image, bot, top,color,-1)
 
    return image

def plot_rectangle(img, l = 10, coords = (100,100)):

    h,w = img.shape[1], img.shape[0]
    bot = (coords[0] - l, coords[1] - l)
    top = (coords[0] + l, coords[1] + l)
    img = cv2.rectangle(img, bot, top,(0,255,0),1)

    return img

def plot_vline(img, x): 
    h,w = img.shape[0], img.shape[1]
    img = cv2.line(img, (x,0),(x,h), color=(0,0,255), thickness=2)

    return img

def plot_hline(img, y): 
    h,w = img.shape[0], img.shape[1]
    
    img = cv2.line(img, (0,y),(w,y), color=(0,0,255), thickness=2)

    return img

def find_markers(img,lines):
    h,w = img.shape[0], img.shape[1]
    m1,m2,m3,m4 = None, None, None, None
    if lines is not None:
        for line in lines: 
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

def distance(coord1,coord2):
    d = -1
    if coord1 and coord2 is not None: 
        x1,y1 = coord1[0], coord1[1]
        x2,y2 = coord2[0], coord2[1]

        x_d = abs(x2 - x1)
        y_d = abs(y2 - y1)
        d = np.sqrt(np.square(y_d) + np.square(x_d))

    return d

def stage(curr_pos, marker_pos, curr_angle ):
    x = None
def main(): 
    count = 1 
    cap = cv2.VideoCapture(url)
    p1,p2,p3,p4,r0,g0 = None, None, None, None, None, None
    to1, to2 = None, None

    initialisation_length = 100
    while cap.isOpened(): 
        
        for i in range(2):
            cap.grab()
                
        ret, frame = cap.retrieve()
        
        fix_frame = rotate_image(undistort(frame), theta)
        h,w = fix_frame.shape[0], fix_frame.shape[1]
        frame3 = fix_frame

        frame2 = filter_crop(fix_frame)
        frame2 = detect_edge(frame2)

        
        if count <= initialisation_length:
            
            line_segments = houghline(frame2, 10)
            frame3, corners, zones = plot_lanes(fix_frame,line_segments)
        
            m1,m2,m3,m4 = find_markers(frame3, corners)
            r1,g1 = find_zones(frame3, zones)
            t1,t2 = tunnel_marker(frame3, line_segments)


            rp = stable_marker(r1, r0, count)
            gp = stable_marker(g1, g0, count)
            
            c1 = stable_marker(m1,p1,count)
            c2 = stable_marker(m2,p2,count)
            c3 = stable_marker(m3,p3,count)
            c4 = stable_marker(m4,p4,count)

            tt1 = stable_marker(t1, to1, count)
            tt2 = stable_marker(t2, to2, count)

        else: 
            c1,c2,c3,c4 = p1,p2,p3,p4
            rp, gp = r0, g0
            tt1, tt2 = to1, to2  

        plot_point(frame3,c1)
        plot_point(frame3,c2)
        plot_point(frame3,c3)
        plot_point(frame3,c4)
        
        plot_point(frame3, rp, color=(100,0,250))
        plot_point(frame3, gp, color=(100,0,250))
        plot_point(frame3, tt1,color= (0,0,250))
        plot_point(frame3, tt2,color= (0,0,250))

        frame3 = plot_hline(frame3, int(h/2))
        frame3 = plot_vline(frame3, int(w/2)) 
        frame3 = plot_hline(frame3, int(h/4))
        
        p1,p2,p3,p4 = c1,c2,c3,c4
        r0,g0 = rp, gp
        to1, to2 = tt1, tt2
        
        cv2.imshow('stream', frame3)

        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

   main()

