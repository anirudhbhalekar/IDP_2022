# IDP_2022

Team 201 - IDP software repository

First Competition Score: 25
Final Competition Score: 60




Description of files included: 

-----------------------------------------------------------------------------------------
*BLE_vari_motor.txt* 

A text file with the arduino code (this is uploaded to the arduino)

-----------------------------------------------------------------------------------------
*stream_test.py*  

This a collection of methods that deals with the raw stream, it has methods for masking, 
undistortion, rotation, and edge detection. 

There are also methods for houghline detection, marker detection for the corners, tunnels,
zones, and plotting those markers. 

There is also a marker stabilisation algorithm and some distance and pose finding functions

-----------------------------------------------------------------------------------------
*detection_tools.py* 

Collection of methods for detection of ArUco markers (this means getting the pose and heading)

Methods for finding and storing position of blue blocks

Methods for detecting orientation relative to markers, there is also a method to encode these as serial commands
(strings)

Initialisation methods to find all markers and store their position after the first N frames 
(this reduces latency)

Update handling for serial communication

GUI initialisation and plotting

-----------------------------------------------------------------------------------------
*vis_tools.py*

Some tools for GUI elements (bounding boxes and arrows) 

-----------------------------------------------------------------------------------------
*transporter.py*

This intialises the target list and then iterates over the robot transport sequence

Responsible for invoking methods from each file to write serial commands to the robot

Conditions for specific targets (procedures) with writing and reading data to the arduino 

Block detection protocol

RUN TO MAKE ROBOT WORK

-----------------------------------------------------------------------------------------
*pyserial.py* 

TEST FILE

Serial testing to see how block detection works


-----------------------------------------------------------------------------------------

*Method Descriptions*

Structure: 

Method name;
Input Arguments; 
Outputs; 
Description; 
-----------------------------------------------------------------------------------------
aruco_display()
Image, corners, ids, rejected
Image

Plots center and arrow heading of the Aruco markers. Takes the corners and ids from the aruco_detection() methods and plots it onto image 
-----------------------------------------------------------------------------------------
aruco_detection()
Img 
Corners, ids 

Uses openCV’s inbuilt aruco-detection algorithm to identify corners and the id of the ArUco marker based on the 4x4 dictionary initialized before
-----------------------------------------------------------------------------------------
return_angle()
Coords0, coords1
angle 

Returns the angle between two coordinates in the image space and applies zone identification to ensure the sign is correct. 
-----------------------------------------------------------------------------------------
get_pose()
Corners, ids
Cx, Cy, angle

Gets the pose and heading of the ArUco marker (based on all 4 corners of the square) 
-----------------------------------------------------------------------------------------
undistort()
Img 
Undistorted_image 

Un-distorts fisheye effect from camera. Camera matrix and distortion coefficients were found through previous testing and then saved. These matrices are then applied to the frame to ensure straight lines remain straight everywhere
-----------------------------------------------------------------------------------------
detect_colour()
Img, upper, lower, ret_colour
Mask

Detects and filters raw feed fed to it in the argument based on lower and upper bounds on the HSV arguments - returns a mask of those colors isolated (this is used in block detection)
-----------------------------------------------------------------------------------------
binary_image()
Img, cutoff = 50
Binary 

Binarizes a grayscale image based on a cutoff defined in the argument. Returns binary image as a matrix of 1s and 0s
-----------------------------------------------------------------------------------------
detect_obj()
Contours
Mean, angle

Detects object position and orientation based on contours fed - this helps in block detection (the mask is converted into a contour map and then fed into this function) 
-----------------------------------------------------------------------------------------
rotate_image()
Image, angle
Result 

Applies a rotation matrix to the input image based on an input angle theta and returns the result
-----------------------------------------------------------------------------------------
find_blue_blocks()
Undistorted_image, return_blue 
Centre_list

Invokes the detec_obj(), mask(), and detect_colour() functions for lower and upper HSV values for blue hues to find the centers of the blue blocks
-----------------------------------------------------------------------------------------
region_of_interest()
Edges
Cropped_edges

Crops the input image based on a percent shaved off the top and bottom
-----------------------------------------------------------------------------------------
detect_edge()
Image 
Edges

Canny detection for edges, used to plot houghlines 
-----------------------------------------------------------------------------------------
mask()
Frame
Mask

Creates a mask based on upper and lower white values to detect the path lines on  the board (used for marker creation)
-----------------------------------------------------------------------------------------
dir_head()
Target_x, target_y, arrow_x, arrow_y, arrow_angle
Distance, rotation

Takes unpacked x and y coordinates of the target and pose of the ArUco marker to give the distance from the target and the angle between the two. This is then fed into the command line function to send serial commands to the arduino 
-----------------------------------------------------------------------------------------
initialize()
Cap, theta, show = False
Rp, gp, c1, c2, c3, c4, tt1 , tt2

Initialisation method - used to find all key features (these are the corners, the tunnel points, and the zone regions)
-----------------------------------------------------------------------------------------
vision()
Cap, theta, phase, block 
Cx, Cy, angle, fix_frame, block

Finds the mean position and orientation of the ArUco markers and plots the same along with plotting the block’s position (if blocks are found) 
-----------------------------------------------------------------------------------------
get_command()
Target, Cx, Cy, angle, thresh = 5, x = 0.6 
Command, distance, rotation

Takes in the target coordinates, the position of the robot marker and its orientation, a threshold to how close the robot must get to the target (thresh) and a damping factor for rotation (0.6) to serialize and write commands to the arduino depending on values of the distance and relative rotation
-----------------------------------------------------------------------------------------
update_handler()
Target, distance, rotation, count
Update, count

Based on the distance and rotation values, it will update the phase the robot is in and the set count (the count being used to time block retrieval and reading) 
-----------------------------------------------------------------------------------------
GUI()
Fix_frame, stable, target, block, Cx, Cy
Fix_frame

Adds all GUI elements (markers, ArUco detection, target arrow and target highlights) onto the display frame

-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

houghlines()
Frame, minL 
Line_segments 

Uses openCVs houghline function to return beginning and end coordinates of all lines detected in the mask. The input frame must be a binary image of the edges detected from the white path lines
-----------------------------------------------------------------------------------------
plot_lanes()
Superimpose, lines, color = (250,100,0), l_width = 2, thresh = 1.01
Line_image, corner_line_arr, zone_line_arr

Plots all the lines with default characteristics onto a superimposed image (the original undistorted frame). It also returns all lines that have a gradient between 0.8 and 1.2 (these are for corner detection) and all vertical lines in the top 25% of the image (the zone markers). It compiles these into arrays 
-----------------------------------------------------------------------------------------
plot_points()
Image, coords, color = (0,255,0) 
Image 

Plots a point set by the coordinates onto the image
-----------------------------------------------------------------------------------------
plot_vline()
Img, x
Img 

Plots a vertical line on a set x-coordinate
-----------------------------------------------------------------------------------------
plot_hline()
Img, y
Img 

Plots a horizontal line on a set y-coordinate
-----------------------------------------------------------------------------------------
find_markers()
img , lines
m1,m2,m3,m4

From all line segments in the corner line array - it finds the midpoints of corresponding markers in different quadrants
-----------------------------------------------------------------------------------------
find_zones()
Img, zone_array
G, r

From the lines in the zone array it will detect the top most coordinate either in the left or right half of the image and correspondingly return it as the green or red zone (this works much more reliably as compared to color detection for zone identification) 
-----------------------------------------------------------------------------------------
tunnel_marker()
Img, all_lines
Max_y, min_y

Finds tunnel markers from all lines - it looks at the right half of the image and at the upper and lower quadrant for discontinuities. 
-----------------------------------------------------------------------------------------
stable_marker()
Curr_markers, prev_markers, count
(xn,yn)

Stability algorithm to prevent confusion - will update the next marker based on the reciprocal of the square root of the distance between the current and previous estimate (this is then modulated by the reciprocal of the count - so at count > 250 there is negligible change). This ensures that after the initialisation period, the marker position is not subject to changing based on interference
-----------------------------------------------------------------------------------------
distance()
Coord1, coord2
d

Returns the distance between two coordinates on the frame
-----------------------------------------------------------------------------------------

