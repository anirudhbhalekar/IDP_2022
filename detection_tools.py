import numpy as np
import cv2

DIM = (1016, 760)
K = np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
D = np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])

def undistort(img):
    #function to remove fisheye from an image taken with the overhead camera
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def detect_colour(img, upper, lower, ret_colour = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if not ret_colour:
        return mask
    else:
        return np.expand_dims(mask, -1) * img

def sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return grad

def edge_detection(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.convertScaleAbs(cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3))
    dy = cv2.convertScaleAbs(cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3))
    grad = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    return grad

def binary_image(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a, binary = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def detect_obj(contours):
    sz = len(contours)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = contours[i,0,0]
        data_pts[i,1] = contours[i,0,1]

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
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

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)       
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

img_path = "1_pink_arrow.jpg"
test_img = cv2.imread(img_path)
undistorted_img = undistort(test_img)
edge = edge_detection(undistorted_img)

def find_green_beam(undistorted_img, edge):
    upper = np.array([31, 270, 255])
    lower = np.array([29, 90,	0])
    single_colour = detect_colour(undistorted_img, upper, lower)
    return detect_objects(single_colour, 100, 10000)

def find_pink_arrow(undistorted_img, edge):
    upper = np.array([165, 255, 255])
    lower = np.array([155, 50,	160])
    single_colour = detect_colour(undistorted_img, upper, lower) * edge
    return detect_objects(single_colour, 100, 10000)

gb_centre, gb_angle = find_green_beam(undistorted_img, edge)
pa_centre, pa_angle = find_pink_arrow(undistorted_img, edge)

print(gb_centre, gb_angle)
print(pa_centre, pa_angle)

rotated = rotate_image(undistorted_img, gb_angle[0] + 180)
cv2.imshow("rotated", rotated)
cv2.waitKey(0)

"""
cv2.imshow('raw', test_img)
cv2.waitKey(0)
cv2.imshow('undistorted', undistorted_img)
cv2.waitKey(0)
cv2.imshow('only green', green_only)
cv2.waitKey(0)
"""