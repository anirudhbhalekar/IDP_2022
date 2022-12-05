# IDP_2022
Team 201 - IDP software repository

First Competition Score: 25
Final Competition Score: 60




Description of files included: 

-----------------------------------------------------------------------------------------
BLE_vari_motor.txt 

A text file with the arduino code (this is uploaded to the arduino)

-----------------------------------------------------------------------------------------
stream_test.py  

This a collection of methods that deals with the raw stream, it has methods for masking, 
undistortion, rotation, and edge detection. 

There are also methods for houghline detection, marker detection for the corners, tunnels,
zones, and plotting those markers. 

There is also a marker stabilisation algorithm and some distance and pose finding functions

-----------------------------------------------------------------------------------------
detection_tools.py 

Collection of methods for detection of ArUco markers (this means getting the pose and heading)

Methods for finding and storing position of blue blocks

Methods for detecting orientation relative to markers, there is also a method to encode these as serial commands
(strings)

Initialisation methods to find all markers and store their position after the first N frames 
(this reduces latency)

Update handling for serial communication

GUI initialisation and plotting

-----------------------------------------------------------------------------------------
vis_tools.py

Some tools for GUI elements (bounding boxes and arrows) 

-----------------------------------------------------------------------------------------
transporter.py

This intialises the target list and then iterates over the robot transport sequence

Responsible for invoking methods from each file to write serial commands to the robot

Conditions for specific targets (procedures) with writing and reading data to the arduino 

Block detection protocol

RUN TO MAKE ROBOT WORK

-----------------------------------------------------------------------------------------
pyserial.py 

TEST FILE

Serial testing to see how block detection works
