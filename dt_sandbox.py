import detection_tools as dt
import vis_tools as vt
import cv2
import numpy as np
import math

img = cv2.imread("2blocks.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = dt.undistort(img)
img = dt.rotate_image(img, 83)
img_final = img

#cv2.imshow("Original", img_final)
#cv2.waitKey(0)

centre_list = dt.find_blue_blocks(img_final)
for centre in centre_list:
    centre = centre[0]
    img_final = vt.plot_rectangle(img_final, 5, centre.astype(int))

#cv2.imshow("Located Blocks", img_final)
#cv2.waitKey(0)

img = cv2.imread("1_pink_arrow.jpg")
img = dt.undistort(img)
img = dt.rotate_image(img, 83)

img = dt.plot_pink_arrow_direction(img, 200, 200)

cv2.imshow("Arrow detected", img)
cv2.waitKey(0)