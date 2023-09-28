#! usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    image = cv2.imread('road.jpg')
    lane_image = np.copy(image)
    canny_image = find_edges(lane_image)
    cv2.imshow("ROI", region_of_interest(canny_image))
    cv2.waitKey(0)

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([
        [(0, height), (1000, height), (650, 200)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    return mask

def find_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 100)
    return canny_image
    

main()