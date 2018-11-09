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

image = ds_list[10].pixel_array

# hist = cv2.calcHist([image],[0],None,[65535],[0,65535])
# plt.plot(hist)
# plt.show(block=False)


image_height, image_width = image.shape
image_norm = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
image_rgb = np.stack([image_norm]*3,axis=-1) #stack for bgr/rgb
image_backup = image_rgb.copy()




def draw_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # image_rgb = image_backup.copy()
        print(f"Value: {image_norm[y,x]} at pixel: {x,y}")
        # cv2.putText(image_rgb, f'test', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (65535,0,0), 2)
        cv2.putText(image_rgb, f'{image_norm[y,x]}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (65535, 0, 0), 2)

cv2.namedWindow("image_norm",cv2.WINDOW_NORMAL)
cv2.imshow("image_norm", image_rgb)
cv2.moveWindow("image_norm",-1440,0)
cv2.setMouseCallback('image_norm', draw_pixel_value)

while True:
    cv2.imshow("image_norm", image_rgb)
    key = cv2.waitKey(1) & 255
    if key == ord('q'):
         break
    if key == ord('r'):
        image_rgb = image_backup.copy()

cv2.destroyAllWindows()








