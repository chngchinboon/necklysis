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

        else:
            skinpoints.append((x, y))

        print(f'Spine Points: {spinepoints}, Skin Points: {skinpoints}')


def drawimage_points(image, points, mode):
    colors = ((0, 65535, 00), (0, 0, 65535))
    for point in points:
        cv2.circle(image, point, 3, colors[mode])


def drawcurve(img, coeff, mode):
    colors = ((65535, 65535, 00), (65535, 0, 65535))
    w, h, _ = img.shape
    f = np.poly1d(coeff)
    x = np.arange(w)
    y = f(x)
    for point in zip(y.astype(int), x.astype(int)):
        cv2.circle(img, point, 1, colors[mode])


def writeoutputtofile(args):
    print('Writing output to file')


def writeimgtofile(args):
    print('Writing image to file')


def fitline(points):
    print('Fitting line')
    # assuming incoming array is [(x1,y1),(x2,y2),....]
    x,y = np.split(points,[1],axis=1)

    fitcoeff = np.polyfit(np.squeeze(x),np.squeeze(y), 5)
    print(fitcoeff)

    return fitcoeff

def plotlineonimg(coeff):
    print('writing line to image')

def write_instructions(image):
    print('Refreshing instructions')

def gui(args):
    global spinepoints, skinpoints
    # do something
    print('running gui')
    image = dcmread(args.filepath).pixel_array
    image_height, image_width = image.shape
    image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    image = np.stack([image] * 3, axis=-1)
    imageshow = image.copy()

    cv2.namedWindow(args.filepath,cv2.WINDOW_NORMAL)
    cv2.imshow(args.filepath, imageshow)
    cv2.setMouseCallback(args.filepath, capmouseclick)
    global mode
    mode = 0
    numspine = 0
    numskin = 0

    key = ''
    while True:
        key = cv2.waitKey(1) & 0xff
        currnumspine = len(spinepoints)
        currnumskin = len(skinpoints)
        if currnumspine != numspine or currnumskin != numskin:
            numspine = currnumspine
            numskin = currnumskin

            # reload original image
            imageshow = image.copy()

            # Draw points
            drawimage_points(imageshow, spinepoints, 0)
            drawimage_points(imageshow, skinpoints, 1)

            # drawcurve
            if numspine > 2:
                # generating line
                print('Num. Spine points >2: attempting to fit line')
                spinecoeff = fitline(spinepoints)
                drawcurve(imageshow, spinecoeff, 0)

            if numskin > 2:
                # generating line
                print('Num. Skin points >2: attempting to fit line')
                spinecoeff = fitline(skinpoints)
                drawcurve(imageshow, spinecoeff, 1)

        if key == ord('q'):
            break
        if key == ord('m'):
            if mode:
                mode = 0
                print('Adding points to spine')
            else:
                mode = 1
                print('Adding points to skin')

        if key == ord('z'):
            if mode == 0 and numspine > 0:
                spinepoints.pop()
            elif mode == 1 and numskin >0:
                skinpoints.pop()
            print(f'Spine Points: {spinepoints}, Skin Points: {skinpoints}')


        if key == ord('r'):
            imageshow = image.copy()

        cv2.imshow(args.filepath, imageshow)

    cv2.destroyAllWindows()
    generatereport(args)

    if args.outputstats:
        writeoutputtofile(args)
    if args.outputimg:
        writeimgtofile(args)


if __name__ ==  "__main__":



    # check input path if file or folder

    # Run gui
    main()

