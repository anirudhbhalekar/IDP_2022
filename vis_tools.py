import detection_tools as dt
import numpy as np
import cv2

def draw_arrow(x, y, angle, img, length = 30):
    angle = angle / 360 * 2 * np.pi
    end_x = int(x + length * np.cos(angle))
    end_y = int(y + length * np.sin(angle))
    return cv2.arrowedLine(img, (int(x), int(y)), (end_x, end_y), (255, 0, 0))

def plot_rectangle(img, l, coords):

    h,w = img.shape[1], img.shape[0]
    bot = (coords[0] - l, coords[1] - l)
    top = (coords[0] + l, coords[1] + l)
    img = cv2.rectangle(img, bot, top,(0,255,0),1)

    return img