import os
import numpy as np
import argparse
from pydicom.filereader import dcmread, read_dicomdir
from glob import glob
from scipy.interpolate import interp1d
from scipy import stats
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

    args = parser.parse_args()

    print(f'Filepath: {args.filepath}')
    print(f'Segmentation type: {args.mode}')
    print(f'Output style: {args.outputargs[0]}')
    print(f'Output paths: {args.outputargs[1:]}')

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
    reportstr = f'Mean: {args[0]}\n' \
                f'Median: {args[1]}\n' \
                f'Mode: {args[2].mode[0]} Count: {args[2].count[0]}'
    print(reportstr)

    return reportstr


def capmouseclick(event, x, y, flags, param):
    global spinepoints, skinpoints, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 0:
            spinepoints.append((x, y))
            sorted(spinepoints, key=lambda k: [k[0], k[1]])

        else:
            skinpoints.append((x, y))
            sorted(skinpoints, key=lambda k: [k[0], k[1]])

        print(f'Spine Points: {spinepoints}, Skin Points: {skinpoints}')


def drawimage_points(image, points, mode):
    colors = ((0, 65535, 00), (0, 0, 65535))
    for point in points:
        cv2.circle(image, point, 3, colors[mode])


def drawmidcurve(img, curve1,curve2):
    colors = (0, 65535, 65535)
    w, h, _ = img.shape

    x1 = curve1[0]
    x2 = curve2[0]

    assert np.array_equal(curve1[1],curve2[1])

    if np.median(x1)> np.median(x2):
        xm = (x1 + x2)/2
        # xd = x1 - x2
    else:
        xm = (x2 + x1)/2
        # xd = x2 - x1
    xd = abs(x1-x2)

    points = zip(xm.astype(int),curve1[1])

    for point in points:
        cv2.circle(img, point, 1, colors)

    return np.mean(xd), np.median(xd), stats.mode(xd)


def drawcurve(img, f, y, mode):
    colors = ((65535, 65535, 00), (65535, 0, 65535))
    w, h, _ = img.shape

    # xmin = x.min()
    # xmax = x.max()
    # dx = xmax-xmin
    # steps = int(dx)
    # x = np.linspace(x.min(), x.max(), steps, endpoint = True )
    # y = f(x)

    # based on assumption image is vertical.

    # ymin = y.min()
    # ymax = y.max()
    # dy = ymax - ymin
    ymin = 0
    ymax = w
    dy = w
    steps = int(dy)
    y = np.linspace(ymin, ymax, steps, endpoint=True)
    x = f(y)

    points = zip(x.astype(int), y.astype(int))
    for point in points:
        cv2.circle(img, point, 1, colors[mode])

    return (x.astype(int), y.astype(int))


def writeoutputtofile(filepath, rpt):
    print(f'Writing output to file: {filepath}')
    with open(filepath,'w') as f:
        print(rpt, file=f)


def writeimgtofile(filepath, img):
    print(f'Writing image to file: {filepath}')
    cv2.imwrite(filepath, img)


def fitline(points):
    print('Fitting line')
    # assuming incoming array is [(x1,y1),(x2,y2),....]
    x,y = np.split(points,[1],axis=1)

    # f = interp1d(np.squeeze(x), np.squeeze(y),  kind='cubic')
    f = interp1d(np.squeeze(y), np.squeeze(x), kind='cubic', fill_value='extrapolate')
    # f_ex = extrap1d(f)

    return f,y


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

    cv2.namedWindow(args.filepath,cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow(args.filepath, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(args.filepath, imageshow)
    cv2.setMouseCallback(args.filepath, capmouseclick)
    global mode
    mode = 0
    numspine = 0
    numskin = 0
    estimateon = 1
    rpt = []

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
            if numspine > 3:
                # generating line
                print('Num. Spine points >2: attempting to fit line')
                spinecoeff, x = fitline(spinepoints)
                spinepts = drawcurve(imageshow, spinecoeff, x, 0)

            if numskin > 3:
                # generating line
                print('Num. Skin points >2: attempting to fit line')
                spinecoeff, x = fitline(skinpoints)
                skinpts = drawcurve(imageshow, spinecoeff, x, 1)


            if numskin > 3 and numspine > 3 and estimateon == 1:
                print('Estimating centroid')
                rpt = drawmidcurve(imageshow, spinepts, skinpts)

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

        if key == ord('e'):
            if estimateon == 0:
                estimateon = 1
            else:
                estimateon = 0

        if key == ord('r'):
            imageshow = image.copy()

        cv2.imshow(args.filepath, imageshow)

    cv2.destroyAllWindows()
    if rpt:
        rpttxt = generatereport(rpt)

        if args.outputstats:
            writeoutputtofile(args.outputargs[1],rpttxt)
        if args.outputimg:
            writeimgtofile(args.outputargs[2],imageshow)


if __name__ ==  "__main__":



    # check input path if file or folder

    # Run gui
    main()

