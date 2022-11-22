import cv2 
import sys 
import numpy as np 


"""
on original feed 

rmse: 3.4264304106452443
camera matrix:
 [[216.94417767   0.         502.48026266]
 [  0.         145.60895573 377.97176059]
 [  0.           0.           1.        ]]

distortion coeffs: [[ 0.04893074  0.03431207 -0.01465341  0.0607378  -0.00279518]]

Rs:
 (array([[ 0.57821595],
       [-0.35696171],
       [-2.00346791]]),)

Ts:
 (array([[ 8.35841569],
       [27.58149642],
       [24.20417851]]),)

"""

url = "http://localhost:8081/stream/video.mjpeg"

cmtx = np.array([[216.94417767,0,502.48026266],[0,145.60895573,377.97176059],[0,0,1]])
dst = np.array([[0.04893074], [0.03431207], [-0.01465341], [0.0607378], [-0.00279518]])

DIM=(1016, 760)
K= np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
D=np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])


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

def get_qr_coords(cmtx, dist, points): 

    qr_edges = np.array([[0,0,0],
                         [0,1,0],
                         [1,1,0],
                         [1,0,0]], dtype = 'float32').reshape((4,1,3))
    
    ret, rvec, tvec = cv2.solvePnP(qr_edges, points, cmtx, dist)
    
    unitv_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    if ret:
        points, jac = cv2.projectPoints(unitv_points, rvec, tvec, cmtx, dist)
        #the returned points are pixel coordinates of each unit vector.
        return points, rvec, tvec

    #return empty arrays if rotation and translation values not found
    else: return [], [], []


def show_axes(cmtx, dist, in_source):
    
    cap = cv2.VideoCapture(in_source)
    qr = cv2.QRCodeDetector()

    while cap.isOpened(): 
        
        for i in range(2):
            cap.grab()


        ret, frame = cap.retrieve()
        ret_qr, points = qr.detect(frame)

        if ret_qr:
            axis_points, rvec, tvec = get_qr_coords(cmtx, dist, points)
            print("DETECTED")
            #BGR color format
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,0,0)]

            #check axes points are projected to camera view.
            if len(axis_points) > 0:
                axis_points = axis_points.reshape((4,2))

                origin = (int(axis_points[0][0]),int(axis_points[0][1]) )

                for p, c in zip(axis_points[1:], colors[:3]):
                    p = (int(p[0]), int(p[1]))

                    #Sometimes qr detector will make a mistake and projected point will overflow integer value. We skip these cases. 
                    if origin[0] > 5*frame.shape[1] or origin[1] > 5*frame.shape[1]:break
                    if p[0] > 5*frame.shape[1] or p[1] > 5*frame.shape[1]:break

                    cv2.line(frame, origin, p, c, 5)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(20)
        if k == 27: break #27 is ESC key.

    cap.release()
    cv2.destroyAllWindows()
 

if __name__ == "__main__": 
    input = url

    show_axes(cmtx, dst, input)


