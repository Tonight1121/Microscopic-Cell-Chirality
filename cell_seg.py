import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import time
import matplotlib.animation as animation
import copy

# data_path = os.chdir('./data')
# cap = cv2.VideoCapture(data_path)

threshold = 0.40
overlapThresh = 0.1
step = 10
mouse_index = 0
img_index = 1
template_size = 160

img_name = 'a{}.bmp'.format(img_index)
board = Image.open('heart/{}'.format(img_name))

log_threshold = 4.5

def img_fill(im_in, n):  # n = binary image threshold
    th, im_th = cv2.threshold(im_in, n, 255, cv2.THRESH_BINARY);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    fill_image = im_th | im_floodfill_inv

    return fill_image

def get_cell_mask(input_patch):
    img = input_patch[26:78, 26:78]
    # cv2.imshow('patch_{}'.format(patch_index), patch_img)
    # cv2.waitKey(0)

    # img = np.log2(img, dtype=np.float32)
    img = cv2.medianBlur(img, 3)
    origin_img = copy.copy(img)
    edges = cv2.Canny(img, 15, 255)

    _, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create an output of all zeroes that has the same shape as the input
    # image
    out = np.zeros_like(img)
    # print(contours.shape)

    # On this output, draw all of the contours that we have detected
    # in white, and set the thickness to be 3 pixels
    cv2.drawContours(out, contours, 0, 255, 1)

    # cv2.floodFill(img, out, (0, 0), (255))

    mask = img_fill(out, 100)
    segment = (mask == 255) * origin_img + (mask == 0) * 0

    # cv2.imwrite('patch_XY15/buffed_{}_closed.jpg'.format(patch_index), mask)
    # cv2.imwrite('patch_XY15/buffed_{}_segment.jpg'.format(patch_index), segment)
    # print('patch_XY15/buffed_{}_closed.jpg written'.format(patch_index))

    return  mask, segment

if __name__ == '__main__':
    patch_index = 42

    for patch_index in range(67):
        img = cv2.imread('patch_XY15/buffed_{}_o.jpg'.format(patch_index), 0)

        ''' This is the core calling '''
        ''' Input a 104*104 patch, output center 52*52 mask and segmentation '''
        mask, segment = get_cell_mask(img)

        cv2.imwrite('patch_XY15/buffed_{}_closed.jpg'.format(patch_index), mask)
        cv2.imwrite('patch_XY15/buffed_{}_segment.jpg'.format(patch_index), segment)
        print('patch_XY15/buffed_{}_closed.jpg written'.format(patch_index))


