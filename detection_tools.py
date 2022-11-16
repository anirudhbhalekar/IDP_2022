import numpy as np
import cv2
import vis_tools as vt
import math
#parameters for camera 1
DIM = (1016, 760)
K = np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
D = np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])

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

def sobel(image):
    #returns sobel filtered image 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return grad

def edge_detection(img):
    #returns map of edges of original image
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.convertScaleAbs(cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3))
    dy = cv2.convertScaleAbs(cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3))
    grad = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    return grad

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
    print(mean)
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

def find_green_beam(undistorted_img):
    #finds the centre green beam from an undistorted image and returns the pixel value of the centre of the beam and it's heading
    upper = np.array([31, 270, 255])
    lower = np.array([29, 90,	0])
    single_colour = detect_colour(undistorted_img, upper, lower)
    centre, angle = detect_objects(single_colour, 100, 10000)
    return centre[0][0][0], centre[0][0][1], angle[0]

def find_pink_arrow(undistorted_img, edge, ret_pink = False):
    #finds the pink arrow from an undistorted image and returns the pixel value of the centre of the arrow and it's heading
    upper = np.array([165, 255, 255])
    lower = np.array([155, 100,	160])
    single_colour = detect_colour(undistorted_img, upper, lower)
    centre, angle = detect_objects(single_colour, 100, 10000)
    if not ret_pink:
        return centre[0][0][0], centre[0][0][1], angle[0]
    return single_colour

def corrected_pink_arrow(undistorted_img, edge, prev_angle):
    x, y, raw_angle = find_pink_arrow(undistorted_img, edge)
    angle_array = np.array([0, 180]) + raw_angle
    
    deltas = np.abs(angle_array - prev_angle)
    big = np.argmax(deltas)
    deltas[big] = 360 - deltas[big]
    quad = np.argmin(deltas)
    #print(deltas)
    angle = angle_array[quad]
    prev_angle = angle
    return x, y, angle, prev_angle


#old blue
#upper = np.array([105, 240, 255])
#lower = np.array([95, 180,	0])

#upper = np.array([120, 180, 255])
#lower = np.array([90, 120,	0])

def find_blue_blocks(undistorted_img, return_blue = False):
    upper = np.array([120, 360, 360])
    lower = np.array([100, 0,	0])

    blues = detect_colour(undistorted_img, upper, lower, False)

    centre_list, _ = detect_objects(blues, 70, 100000)
    if return_blue:
        return centre_list, blues
    else:
        return centre_list

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

def no_filter_crop(image): 
    img = region_of_interest(detect_edge(mask(image)))
    return img

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

def route(x, y, target_x, target_y):
    dx = target_x - x
    dy = target_y - y
    distance = np.sqrt(dx**2 + dy**2)
    heading = np.arctan(dx / dy) * 180 / np.pi
    if dy < 0 and dx >= 0:
        heading = 180 + heading
    if dy < 0 and dx < 0:
        heading = 180 + heading
    if dy >= 0 and dx < 0:
        heading = 360 + heading
    return distance, heading 

def pink_arrow_line(img, prev_angle):
    edge = edge_detection(img)
    arrow_x, arrow_y, arrow_angle, prev_angle = corrected_pink_arrow(img, edge, prev_angle)
    img = vt.draw_arrow(arrow_x, arrow_y, arrow_angle, img, 30)
    arrow = np.array([arrow_x, arrow_y])
    return img, arrow

def plot_pink_arrow_direction(img, target_x, target_y, prev_angle):
    target = (target_x, target_y)
    edge = edge_detection(img)
    arrow_x, arrow_y, arrow_angle, prev_angle = corrected_pink_arrow(img, edge, prev_angle)
    arrow = np.array([arrow_x, arrow_y])

    img = vt.draw_arrow(arrow_x, arrow_y, arrow_angle, img, 30)
    
    img = vt.plot_rectangle(img, 10, target)

    delta = target - arrow
    distance = np.sqrt(np.sum(np.square(delta)))
    angle = math.atan(delta[1] / delta[0]) * 180 / math.pi

    if delta[0] < 0:
        angle += 180

    img = vt.draw_arrow(arrow_x, arrow_y, angle, img, distance)

    rotation = 720 + angle - arrow_angle
    rotation = rotation % 360
    return img, distance, rotation, prev_angle

def arrow_to_blocks(img, prev_angle):
    block_locs = find_blue_blocks(img)
    first_block = block_locs[1][0]
    block_x, block_y = int(first_block[0]), int(first_block[1])
    return plot_pink_arrow_direction(img, block_x, block_y, prev_angle)

def arrow_to_all_blocks(img, prev_angle):
    block_locs = find_blue_blocks(img)
    img, arrow = pink_arrow_line(img, prev_angle)
    for block in block_locs:
        delta = block[0] - arrow
        distance = np.sqrt(np.sum(np.square(delta)))
        angle = math.atan(delta[1] / delta[0]) * 180 / math.pi

        if delta[0] < 0:
            angle += 180

        img = vt.draw_arrow(arrow[0], arrow[1], angle, img, distance)
    return img


"""

img_path = "1_pink_arrow.jpg"
test_img = cv2.imread(img_path)
undistorted_img = undistort(test_img)
cv2.imshow("Undistorted", undistorted_img)
cv2.waitKey(0)
#edge = edge_detection(undistorted_img)

gb_x, gb_y, gb_angle = find_green_beam(undistorted_img, edge)
pa_x, pa_y, pa_angle = find_pink_arrow(undistorted_img, edge)

print(gb_x, gb_y, gb_angle)
print(pa_x, pa_y, pa_angle)

rotated = rotate_image(undistorted_img, gb_angle + 180)
cv2.imshow("rotated", rotated)
cv2.waitKey(0)

cv2.imshow('raw', test_img)
cv2.waitKey(0)
cv2.imshow('undistorted', undistorted_img)
cv2.waitKey(0)
cv2.imshow('only green', green_only)
cv2.waitKey(0)
"""