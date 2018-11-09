import matplotlib.pyplot as plt
import pydicom
import os
from pydicom.filereader import dcmread, read_dicomdir
from glob import glob
import cv2
import numpy as np
import shutil

# folder specific
# dicom_dir = r"E:\BTSynchSGH\datasets\TCGA-HNSC\TCGA-BA-4074\02-28-1994-CT NECK SOFT TISSUE  W CONTR-40819\3-NeckST MPR  3.0  B30s-75963"

# fps = glob(os.path.join(dicom_dir,"*.dcm"))

fps = glob("E:\BTSynchSGH\datasets\TCGA-HNSC\**\*.dcm", recursive=True)

# grab only those with 'sag'

sagonly = [file for file in fps if 'sag' in file]

sagfoldernames = list(set([os.path.dirname(file) for file in sagonly]))

fps=[]

for folder in sagfoldernames:
    filelist = [file for file in sagonly if folder in file]
    # grab middle files
    fps.append(filelist[int(len(filelist)/2)])

# fps = glob(r"E:\BTSynchSGH\datasets\necklysis\input\dicom\*.dcm")
# fps = sagonly
fps = [r"E:\BTSynchSGH\datasets\necklysis\input\mr\DICOM\18110910\05350000\49224840"]

# transfer subset to folder
# savelist = [0,2,4,10,12,18,20,21,22,23,24,25,26]
# for idx,file_idx in enumerate(savelist):
#     shutil.copy(fps[file_idx], os.path.join(r"E:\BTSynchSGH\datasets\necklysis\input\dicom", f"{idx}.dcm"))

# ds_list = [dcmread(filename) for filename in fps]
# plt.imshow(ds_list[3].pixel_array, cmap=plt.cm.bone)
num_images = len(fps)
key = ''
counter = 0
current_window_name = ''

while counter < num_images:

    image = dcmread(fps[counter]).pixel_array

    image = np.stack([image] * 3, axis=-1)  # stack for bgr/rgb
    image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    cv2.putText(image, f'{counter}/{num_images}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (65535, 65535, 65535), 2)  # Put date on the frame

    new_window_name = f"{os.path.dirname(fps[counter])}"

    if current_window_name == '':
        current_window_name = new_window_name
    elif current_window_name != new_window_name:
        cv2.destroyWindow(current_window_name)
        current_window_name = new_window_name

    cv2.imshow(f"{os.path.dirname(fps[counter])}", image)

    key = cv2.waitKey(1) & 0xff

    if key == ord('n'):
        counter += 1

    if key == ord('b'):
        counter -= 1

    if key == ord('q'):
        break

    if key == ord('s'):
        shutil.copy(fps[counter], os.path.join(r"E:\BTSynchSGH\datasets\necklysis\input\dicom\new", f"{counter}.dcm"))

cv2.destroyAllWindows()



