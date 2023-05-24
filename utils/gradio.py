import cv2
import numpy as np
import torch
from adain_transfer import *

def add_bbox(image, box, linewidth=5):
    """ Draw a bbox above the image
        Since this overwrites the pixels, use only for interactive outputs on gradio.Image
    """
    H, W, C = image.shape
    new_img = np.copy(image)
    
    # coordinates of the bbox's inner corners 
    x1 = min(box[0][0], box[1][0])
    x2 = max(box[0][0], box[1][0])
    y1 = min(box[0][1], box[1][1])
    y2 = max(box[0][1], box[1][1])
    
    # coordinates of the bbox's outer corners
    outer_x1 = max(0, (x1 - linewidth))
    outer_y1 = max(0, (y1 - linewidth))
    outer_x2 = min(W, (x2 + linewidth))
    outer_y2 = min(H, (y2 + linewidth))
    
    # overwrite the bbox's outline pixels with [255,0,0] (red)
    new_img[outer_y1:outer_y2, outer_x1:outer_x2, 0] = 255
    new_img[outer_y1:outer_y2, outer_x1:outer_x2, 1] = 0
    new_img[outer_y1:outer_y2, outer_x1:outer_x2, 2] = 0
    new_img[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :]

    return new_img

def add_point(image, point, size=8):
    """ Draw a small square above the image
        Since this overwrites the pixels, use only for interactive outputs on gradio.Image
    """
    H, W, C = image.shape
    new_img = np.copy(image)
    
    #coordinates of the square's corner
    x1 = max(0, (point[0] - size // 2))
    y1 = max(0, (point[1] - size // 2))
    x2 = min(W, (point[0] + size // 2))
    y2 = min(H, (point[1] + size // 2))
    
    # overwrite around the point with [255,0,0]
    new_img[y1:y2, x1:x2, 0] = 255
    new_img[y1:y2, x1:x2, 1] = 0
    new_img[y1:y2, x1:x2, 2] = 0
    
    return new_img
