import cv2
import numpy as np
import matplotlib.pyplot as plt


def gray_and_otsu(image):
    # convert image to 1 channel array
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binarize the image
    otsu, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return image

def minmax255(image):
    # scale pixels to [0,255]
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.astype(np.uint8)        

def postprocess(image):
    image = minmax255(image)
    image = gray_and_otsu(image)
    return image

# zero the value of masked pixels
def remove_background(image, mask):
    dst = np.copy(image)
    H, W, C = image.shape
    mask = mask[:, :, np.newaxis].repeat(C, -1)
    dst[mask==False] = 0
    return dst

def zero_padding(image, shape): 
    H, W, C = image.shape
    dh = shape[0] - H
    dw = shape[1] - W
    
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# for background-removed image
def center_object(image, mask, ratio=2):
    # find the bbox of the mask
    row, col = np.nonzero(mask)
    top, bottom = row.min(), row.max()
    left, right = col.min(), col.max()
    
    # crop the bbox and pad to the target shape
    obj_image = image[top:bottom, left:right, :]
    H, W, C = obj_image.shape
    H = int(H * ratio)
    W = int(W * ratio)
    return zero_padding(obj_image, [H, W])

# draw objects above matplotlib images
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    