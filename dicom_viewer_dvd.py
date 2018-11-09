import matplotlib.pyplot as plt
import pydicom
import os
from pydicom.filereader import dcmread, read_dicomdir
from glob import glob
import cv2
import numpy as np
import shutil

filepath = r"E:\BTSynchSGH\datasets\necklysis\input\mr\DICOMDIR"
ds = read_dicomdir(filepath)
base_dir = os.path.dirname(filepath)
fps = []
for record in ds.DirectoryRecordSequence:
    if record.DirectoryRecordType == "IMAGE":
    # Extract the relative path to the DICOM file
        fps.append(os.path.join(base_dir,*record.ReferencedFileID))

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



