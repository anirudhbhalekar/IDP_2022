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

def binary_image(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey', grey)
    cv2.waitKey(0)
    a, bw = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary', bw)
    cv2.waitKey(0)
    print(a)

def detect_obj(binary_img):
    j = 5
    return j


img_path = "1_pink_arrow.jpg"
test_img = cv2.imread(img_path)
undistorted_img = undistort(test_img)
objects = binary_image(test_img)

"""
#green
upper = np.array([31, 270, 255])
lower = np.array([29, 90,	0])

green_only = detect_colour(undistorted_img, upper, lower, True)

cv2.imshow('raw', test_img)
cv2.waitKey(0)
cv2.imshow('undistorted', undistorted_img)
cv2.waitKey(0)
cv2.imshow('only green', green_only)
cv2.waitKey(0)
"""