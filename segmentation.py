import matplotlib.pyplot as plt
# import pydicom
import os
from pydicom.filereader import dcmread, read_dicomdir
from glob import glob
import cv2
import numpy as np

cv2.destroyAllWindows()
# window prop
screensize = ((-1440,0),(0,900))
screenwidth = screensize[0][1]-screensize[0][0]
screenheight = screensize[1][1]-screensize[1][0]

headertop= 30
headerbottom = 8
headerside = 8

n = 3
m = 2
windowwidth = int((screenwidth - n * headerside*2)/ n)
windowheight = int((screenheight - m * (headertop + headerbottom)) /m)


# input directory
dicom_dir = r"E:\BTSynchSGH\datasets\necklysis\input\dicom"

fps = glob(os.path.join(dicom_dir,"*.dcm"))

ds_list = [dcmread(filename) for filename in fps]


# select image
image = ds_list[10].pixel_array
# image details
image_height, image_width = image.shape

# image pre-processing
image_norm = cv2.normalize(image, dst=None, alpha=0, beta=65536, norm_type=cv2.NORM_MINMAX) # so that can see better
image_norm_uint8 = cv2.convertScaleAbs(image_norm)

min_head_thresh = 10000
max_head_thresh = 65535

# get outline of head
ret, image_thresh = cv2.threshold(image_norm,min_head_thresh, max_head_thresh, cv2.THRESH_TOZERO)
image_thresh_uint8 = cv2.convertScaleAbs(image_thresh)
image_canny = cv2.Canny(image_thresh_uint8,100,150)

# get contour
im2, contours, hierarchy = cv2.findContours(image_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
image_norm_3chan = np.stack([image_norm]*3,axis=-1)


# get largest contour
perimeter = [cv2.arcLength(cnt,True) for cnt in contours]
idx_max = np.argmax(np.array(perimeter))
image_contours = cv2.drawContours(image_norm_3chan.copy(), [contours[idx_max]], 0, (0,65535,0), 3)

# display process images

# original image
cv2.namedWindow("image_norm",cv2.WINDOW_NORMAL)
cv2.moveWindow("image_norm",screensize[0][0],0)
cv2.resizeWindow("image_norm",(windowwidth,windowheight))
cv2.imshow("image_norm", image_norm)

# canny
cv2.namedWindow("image_canny",cv2.WINDOW_NORMAL)
cv2.imshow("image_canny", image_canny)
cv2.resizeWindow("image_canny",(windowwidth,windowheight))
cv2.moveWindow("image_canny",screensize[0][0]+(windowwidth+headerside*2),0)

# contours
cv2.namedWindow("contours",cv2.WINDOW_NORMAL)
cv2.imshow("contours", image_contours)
cv2.resizeWindow("contours",(windowwidth,windowheight))
cv2.moveWindow("contours",screensize[0][0]+(windowwidth+headerside)*2,0)


# cv2.waitKey(1)
# cv2.destroyAllWindows()
