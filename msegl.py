import os
import numpy as np
from pydicom.filereader import dcmread, read_dicomdir
from glob import glob
from scipy.interpolate import interp1d
from scipy import stats
import cv2


class imageseg:
    def __init__(self, input, mode, output):
        # run time parameters
        self.input_path = input
        self.seg_mode = mode
        self.output_style = output[0]
        self.process_run_time_param(output)

        # container for data
        self.spine_points = []
        self.spine_contour = ()
        self.num_spine_points = 0
        self.skin_points = []
        self.skin_contour = ()
        self.num_skin_points = 0

        # Gui color definitions
        self.gui_point_color_spine = (0, 65535, 00)
        self.gui_contour_color_spine = (65535, 65535, 00)
        self.gui_point_color_skin = (0, 0, 65535)
        self.gui_contour_color_skin = (65535, 0, 65535)
        self.gui_contour_color_mid = (0, 65535, 65535)

        # Gui settings
        self.point_select_mode = 'Spine'
        self.window_name = self.input_path
        self.gui_mid_on = True
        self.run_gui = True

        # read image
        self.image_height = None
        self.image_width = None
        self.read_image()

        # container for calculations
        self.mid_curve_mean = None
        self.mid_curve_median = None
        self.mid_curve_mode = None

        # output
        self.report_str = ''


    def process_run_time_param(self, output):
        if self.output_style == 'img':
            self.output_img = True
            self.output_stats = False
            print('Outputting only image')

        elif self.output_style == 'stats':
            self.output_img = False
            self.output_stats = True
            print('Outputting only stats')

        elif self.output_style == 'all':
            self.output_img = True
            self.output_stats = True
            print('Outputting everything')

        else:
            self.output_img = True
            self.output_stats = True
            print('No output')

        if self.output_img:
            img_extensions_to_check = ('.jpg', '.tif', '.png')
            self.output_img_path = [path for path in output if path.endswith(img_extensions_to_check)]
            if len(self.output_img_path) > 1:
                raise ValueError("More than one img path included")
            else:
                self.output_img_path = self.output_img_path[0]

        if self.output_stats:
            txt_extensions_to_check = ('.csv', '.txt', '.xls')
            self.output_stats_path = [path for path in output if path.endswith(txt_extensions_to_check)]
            if len(self.output_stats_path) > 1:
                raise ValueError("More than one report path included")
            else:
                self.output_stats_path = self.output_stats_path[0]


    # input
    def read_image(self, *input_path):
        if input_path:
            print('Overriding image')
            self.input_path = input_path[0]

        self.image = dcmread(self.input_path).pixel_array  # retain original image in case we need to reset
        self.image_height, self.image_width = self.image.shape

        # preprocess
        self.image = cv2.normalize(self.image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        self.image = np.stack([self.image] * 3, axis=-1) # convert to 3 channel so that can display contours
        self.image_show = self.image.copy()  # duplicate image for display

        print('Image loaded')


    # calculations
    def calculate_mid_curve(self):
        x1 = self.spine_contour[0]
        x2 = self.skin_contour[0]

        assert np.array_equal(self.spine_contour[1], self.skin_contour[1])

        if np.median(x1)> np.median(x2):
            xm = (x1 + x2)/2
            # xd = x1 - x2
        else:
            xm = (x2 + x1)/2
            # xd = x2 - x1
        xd = abs(x1-x2)

        self.mid_curve = zip(xm.astype(int), self.spine_contour[1])

        self.mid_curve_mean = np.mean(xd)
        self.mid_curve_median = np.median(xd)
        self.mid_curve_mode = stats.mode(xd).mode[0]


    def fit_line(self, points):
        print('Fitting line')
        # assuming incoming array is [(x1,y1),(x2,y2),....]
        x, y = np.split(points, [1], axis=1)
        f = interp1d(np.squeeze(y), np.squeeze(x), kind='cubic', fill_value='extrapolate')

        return f, y


    # Gui
    def cap_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.point_select_mode == 'spine':
                self.spine_points.append((x, y))
                sorted(self.spine_points, key=lambda k: [k[0], k[1]])
                self.num_spine_points = len(self.spine_points)

            else:
                self.skin_points.append((x, y))
                sorted(self.skin_points, key=lambda k: [k[0], k[1]])
                self.num_skin_points = len(self.skin_points)

            print(f'Spine Points: {self.spine_points}, Skin Points: {self.skin_points}')


    def draw_points_on_image(self, points, color):
        for point in points:
            cv2.circle(self.image_show, point, 3, color)


    def draw_contour_on_image(self, points, color):
        for point in points:
            cv2.circle(self.image_show, point, 1, color)


    def draw_func_within_image(self, func, color):
        ymin = 0
        ymax = self.image_height
        dy = self.image_height # presumed top down hence using Y axis. may need changing for generalization
        steps = int(dy)
        y = np.linspace(ymin, ymax, steps, endpoint=True)
        x = func(y)

        points = zip(x.astype(int), y.astype(int))
        self.draw_contour_on_image(points, color)

        return (x.astype(int), y.astype(int)) #?????? why needed?


    def write_instructions_on_gui(self):
        print('Refreshing instructions')


    def build_gui(self, *custom_window_name):
        if custom_window_name:
            self.window_name = custom_window_name[0]

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow(args.filepath, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(self.window_name, self.image_show)
        cv2.setMouseCallback(self.window_name, self.cap_mouse_click)
        print("Gui Loaded")


    def gui(self):
        # key = ''
        gui_num_spine_points = 0
        gui_num_skin_points = 0
        while self.run_gui:
            key = cv2.waitKey(1) & 0xff
            if gui_num_spine_points != self.num_spine_points or gui_num_skin_points != self.num_skin_points:
                # new points added in previous iteration. need to update image.
                gui_num_spine_points = self.num_spine_points
                gui_num_skin_points = self.num_skin_points

                # reload original image
                self.reset_image()

                # Draw points
                self.draw_points_on_image(self.spine_points, self.gui_point_color_spine)
                self.draw_points_on_image(self.skin_points, self.gui_point_color_skin)

                # drawcurve
                if self.num_spine_points > 3:
                    # generating line
                    print('Num. Spine points >2: attempting to fit line')
                    self.spine_func, _ = self.fit_line(self.spine_points)
                    self.spine_contour = self.draw_func_within_image(self.spine_func, self.gui_contour_color_spine)

                if self.num_skin_points > 3:
                    # generating line
                    print('Num. Skin points >2: attempting to fit line')
                    self.skin_func, _ = self.fit_line(self.skin_points)
                    self.skin_contour = self.draw_func_within_image(self.skin_func, self.gui_contour_color_skin)

                if self.num_spine_points > 3 and self.num_skin_points > 3 and self.gui_mid_on:
                    print('Estimating centroid')
                    self.calculate_mid_curve()
                    self.draw_points_on_image(self.mid_curve, self.gui_contour_color_mid)

                cv2.imshow(self.window_name, self.image_show)

            self.process_gui_keys(key)


    def process_gui_keys(self, key):
        if key == ord('q'):
            self.run_gui = False

        if key == ord('m'):
            if self.point_select_mode == 'spine':
                self.point_select_mode = 'skin'
                print('Adding points to skin')
            else:
                self.point_select_mode = 'spine'
                print('Adding points to spine')

        if key == ord('z'):
            if self.point_select_mode == 'spine' and self.num_spine_points > 0:
                self.spine_points.pop()
                self.num_spine_points = len(self.spine_points)
            elif self.point_select_mode == 'skin' and self.num_skin_points > 0:
                self.skin_points.pop()
                self.num_skin_points = len(self.skin_points)
            print(f'Spine Points: {self.spine_points}, Skin Points: {self.skin_points}')

        if key == ord('e'):
            if self.gui_mid_on:
                self.gui_mid_on = False
            else:
                self.gui_mid_on = True

        if key == ord('r'):
            self.reset_image()


    def reset_image(self):
        self.image_show = self.image.copy()
        print('Resetted image')


    # output
    def generate_report_str(self):
        print('Generating report')
        self.report_str = f'Mean: {self.mid_curve_mean}\n' \
                          f'Median: {self.mid_curve_median}\n' \
                          f'Mode: {self.mid_curve_mode}'
        print(self.report_str)


    def write_output_to_file(self):
        print(f'Writing output to file: {self.output_stats_path}')
        with open(self.output_stats_path, 'w') as file:
            print(self.report_str, file=file)


    def write_img_to_file(self):
        print(f'Writing image to file: {self.output_img_path}')
        cv2.imwrite(self.output_img_path, self.image_show)


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

    gui = imageseg(args.filepath, args.mode, args.outputargs)
    gui.build_gui()
    gui.gui()

    if gui.output_stats:
        gui.generate_report_str()

        if gui.output_stats:
            gui.write_output_to_file()
        if gui.output_img:
            gui.write_img_to_file()


if __name__ ==  "__main__":
    import argparse
    main()


