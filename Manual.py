import os
import numpy as np
import argparse
from pydicom.filereader import dcmread, read_dicomdir
from glob import glob
import cv2

mode = 0
spinepoints = []
skinpoints = []

def main():
    # define command line arguments
    parser = argparse.ArgumentParser(description='Process input image.')
    parser.add_argument('filepath', help='Path to image file')
    parser.add_argument('--mode', dest='mode', default='manual',
                        help='segmentation type (manual/auto)')
    parser.add_argument('--output', dest='outputargs', nargs='+',
                        default=['all','\\'], help='img/stats/all [filenames]')
    # parser.add_argument('--outputpaths',  required=False,
    #                     default='\\', help='path to output')
    # parser.add_argument('outputtxtpath', type=argparse.FileType('w'), nargs='?',
    #                     default='\\report.txt', help='path to outputfile')

    args = parser.parse_args()

    print(f'Filepath: {args.filepath}')
    print(f'Segmentation type: {args.mode}')
    print(f'Output style: {args.outputargs[0]}')
    print(f'Output paths: {args.outputargs[1:]}')

    # if args.ptype == 'manual':
    if args.outputargs[0] == 'img':
        args.outputimg = True
        args.outputstats = False
        print('Outputting only image')


    elif args.outputargs[0] == 'stats':
        args.outputimg = False
        args.outputstats = True
        print('Outputting only stats')


    else:
        args.outputimg = True
        args.outputstats = True
        print('Outputting everything')


    gui(args)

def generatereport(args):
    print('Generating report')

def capmouseclick(event, x, y, flags, param):
    global spinepoints, skinpoints, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 0:
            spinepoints.append((x, y))
            image_points(param, spinepoints, 0)
        else:
            skinpoints.append((x, y))
            image_points(param, skinpoints, 1)

    print(f'Spine Points: {spinepoints}, Skin Points: {skinpoints}')

def image_points(image, points, mode):
    colors = ((0,65535,00),(0,0,65535))
    for point in points:
        cv2.circle(image, point, 3, colors[mode])

def gui(args):
    # do something
    print('running gui')
    image = dcmread(args.filepath).pixel_array
    image_height, image_width = image.shape
    image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    image = np.stack([image] * 3, axis=-1)
    imageshow = image.copy()

    cv2.namedWindow(args.filepath,cv2.WINDOW_NORMAL)
    cv2.imshow(args.filepath, imageshow)
    cv2.setMouseCallback(args.filepath, capmouseclick, imageshow)
    mode = True

    key = ''
    while True:
        key = cv2.waitKey(1) & 0xff
        cv2.imshow(args.filepath,imageshow)



        if key == ord('q'):
            break
        if key == ord('m'):
            if mode:
                mode = 0
            else:
                mode = 1
        if key == ord('z'):
            if mode == 0:
                spinepoints.pop()
                imageshow = image.copy()
            else:
                skinpoints.pop()
                imageshow = image.copy()
            break

    cv2.destroyAllWindows()
    generatereport(args)

if __name__ ==  "__main__":



    # check input path if file or folder

    # Run gui
    main()

