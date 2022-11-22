import numpy as np
import time
import cv2
import detection_tools as dt

url = "http://localhost:8081/stream/video.mjpeg"


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

def return_angle(coords0, coords1, thresh = 0.01): 
    x0,y0 = coords0[0], coords0[1]
    x1,y1 = coords1[0], coords1[1]
    
    
    if abs(x1-x0) > thresh:
        tan_theta = (y1-y0)/(x1-x0)
        theta = np.arctan(tan_theta)
        theta = theta/np.pi * 180
    
    else: 
        if y1 > y0: 
            theta = 90
        else: 
            theta = -90

    return theta
def aruco_display(corners, ids, rejected, image): 
    if len(corners) > 0: 
        print("Detected")
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4,2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            print(corners)

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0,255,0), 3)
            cv2.line(image, topRight, bottomRight, (0,255,0), 3)
            cv2.line(image, bottomRight, bottomLeft, (0,255,0), 3)
            cv2.line(image, bottomLeft, topLeft, (0,255,0), 3)

            topMid = (int((topRight[0] + topLeft[0])/2.0), int((topRight[1] + topLeft[1])/2.0))
            bottomMid = (int((bottomLeft[0] + bottomRight[0])/2.0), int((bottomLeft[1] + bottomRight[1])/2.0))

            cX = int((topLeft[0] + bottomRight[0])/2.0)
            cY = int((topLeft[1] + bottomRight[1])/2.0)
            cv2.circle(image, (cX,cY), 8, (0,0,255), -1)

            cv2.arrowedLine(img, topMid, bottomMid, (255,0,0), 2)
            
            angle = return_angle(bottomMid, topMid)
            cv2.putText(image, str(angle), (topMid[0], topMid[1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
            print("[Inference] ArUco marker ID: {}".format(markerID))

            return image



aruco_type = "DICT_4X4_250"

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()


cap = cv2.VideoCapture(url)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened(): 
    ret, img = cap.read() 
    img = dt.undistort(img)
    
    h, w, _ = img.shape

    width = 1000
    height = int(width*(h/w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC )
   
    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

    detected_markers = aruco_display(corners, ids, rejected, img)

    try: 
        cv2.imshow("image", detected_markers)
    except: 
        cv2.imshow("image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break 

cv2.destroyAllWindows()
cap.release()
