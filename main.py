import matplotlib.pyplot as plt
import pydicom
import os
from pydicom.filereader import dcmread, read_dicomdir
from glob import glob
import cv2
import numpy as np


dicom_dir = r"E:\BTSynchSGH\datasets\necklysis\input\dicom"
headertop= 10
headerbottom = 8
headerside = 10

fps = glob(os.path.join(dicom_dir,"*.dcm"))

ds_list = [dcmread(filename) for filename in fps]

image = ds_list[1].pixel_array
image_height, image_width = image.shape
image_norm = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

image_norm_uint8 = cv2.convertScaleAbs(image_norm)
image_canny = cv2.Canny(image_norm_uint8,100,150)

image_uint8 = cv2.convertScaleAbs(image)
image_canny2 = cv2.Canny(image_uint8,100,150)








cv2.namedWindow("image_norm",cv2.WINDOW_NORMAL)
cv2.imshow("image_norm", image_norm)
cv2.moveWindow("image_norm",-1440,0)

cv2.namedWindow("image_norm_uint8",cv2.WINDOW_NORMAL)
cv2.imshow("image_norm_uint8", image_norm_uint8)
cv2.moveWindow("image_norm_uint8",-1440+image_width+headerside,0)

cv2.namedWindow("image_canny",cv2.WINDOW_NORMAL)
cv2.imshow("image_canny", image_canny)
cv2.moveWindow("image_canny",-1440+(image_width+headerside)*2,0)

cv2.namedWindow("image_uint8",cv2.WINDOW_NORMAL)
cv2.imshow("image_uint8", image_uint8)
cv2.moveWindow("image_uint8",-1440+image_width+headerside,image_height+headertop+headerbottom)

cv2.namedWindow("image_canny2",cv2.WINDOW_NORMAL)
cv2.imshow("image_canny2", image_canny2)
cv2.moveWindow("image_canny2",-1440+(image_width+headerside)*2,image_height+headertop+headerbottom)



