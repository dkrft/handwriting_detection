import cv2
import numpy as np
from scipy import signal
from hwdetect.preprocessor.preprocessor import Preprocessor
# from hwdetect.utils import show

class Hough(Preprocessor):
    def __init__(self, bw_threshold = 214,
                       density_threshold_2 = 248,
                       density_threshold_1 = 0.174,
                       filter_size_multiplicator_1 = 26,
                       filter_size_multiplicator_2 = 57,
                       vertical_filter_size_1 = 0.32,
                       vertical_filter_size_2 = 0.145,
                       # hough:
                       min_line_length = 114,
                       max_line_gap = 4,
                       long_line_factor = 3,
                       epsilon_v_1 = 78.1,
                       epsilon_h_1 = 7.37,
                       epsilon_v_2 = 12.6,
                       epsilon_h_2 = 1.14,
                       min_samples_1 = 5,
                       min_samples_2 = 1):
        self.bw_threshold = bw_threshold
        self.density_threshold_2 = density_threshold_2
        self.density_threshold_1 = density_threshold_1
        self.filter_size_multiplicator_1 = filter_size_multiplicator_1
        self.filter_size_multiplicator_2 = filter_size_multiplicator_2
        self.vertical_filter_size_1 = vertical_filter_size_1
        self.vertical_filter_size_2 = vertical_filter_size_2
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.long_line_factor = long_line_factor
        self.epsilon_v_1 = epsilon_v_1
        self.epsilon_h_1 = epsilon_h_1
        self.epsilon_v_2 = epsilon_v_2
        self.epsilon_h_2 = epsilon_h_2
        self.min_samples_1 = min_samples_1
        self.min_samples_2 = min_samples_2
    """The parameters are already somewhat optimized.
    
    Finds vertical lines in the image using hough transform,
    especially on machine written text, then clusters them:

    1. do dbscan with a wide rectangular epsilon such that
    vertical lines are considered neighbors when they are on
    a similar height

    2. do dbscan on the cores from step 1, which further removes
    noise (noise = vertical lines on handwriting for example)

    afterwards, the image is set to white when there is a
    vertical line, which removes machine writing"""


    # rectangular density based clustering
    # uses cityblock distance and two epsilons
    # one for each dimension
    def rdbscan(self, points, eps_x, eps_y, *args, **kwargs):
        """points is an array of [x1, y1], [x2, y2], ... pairs.
        Returns label id for each point."""

        # do the following thing on a copy and use np arrays:
        points = np.array(points, dtype=float)

        # Distance matrix contains only one
        # value per datapoint pair, so to
        # get a rectangular epsilon, the
        # ratio between eps_x and eps_y needs
        # to be encoded in the distance values.
        # Divide x or y value such that the
        # rectangular cityblock epsilon can
        # be calculated by a squared epsilon.
        eps = 0
        if eps_x > eps_y:
            points[:,0] /= (eps_x / eps_y)
            eps = eps_y
        elif eps_x < eps_y:
            points[:,1] /= (eps_y / eps_x)
            eps = eps_x
        else:
            eps = eps_x

        db = DBSCAN(eps=eps, *args, metric='cityblock', **kwargs).fit(points)
        labels = db.labels_

        return labels


    def show_lines(self, img, lines1, lines2=None):
        img = img.copy() >> 1
        for x in range(0,  len(lines1)):
            for x1, y1, x2, y2 in lines1[x]:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if not lines2 is None:
            for x in range(0,  len(lines2)):
                for x1, y1, x2, y2 in lines2[x]:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        show(img)


    def blur(self, img, kernel_size, height_factor=1):
        """set height_factor to 0 for a vertical-only filter"""
        
        # make a gaussian kernel filter by calculating the cartesian
        # product of two gaussian signals
        # also check for n elements and throw idk error before
        # screwing up memory
        # note, that resize want's a np array of type float

        gauss_1D = signal.gaussian(kernel_size, std=kernel_size/5)
        height = int(max(1, np.ceil(kernel_size*height_factor)))
        a = gauss_1D[None,:]
        b = cv2.resize(gauss_1D[:,None], (1, height))
        kernel = a * b # cartesian product

        kernel /= kernel.sum()
        # apply kernel
        img_blur = cv2.filter2D(img, -1, kernel)

        # the blur will also remove the contrast, renormalize
        # img_blur -= img_blur.min()
        # img_blur = img_blur / img_blur.max()
        
        return img_blur


    def filter(self, img, verbose=False, repititions=1):

        # ---- hyperparameters ---- 
        bw_threshold = self.bw_threshold
        density_threshold_2 = self.density_threshold_2
        density_threshold_1 = self.density_threshold_1
        filter_size_multiplicator_1 = self.filter_size_multiplicator_1
        filter_size_multiplicator_2 = self.filter_size_multiplicator_2
        vertical_filter_size_1 = self.vertical_filter_size_1
        vertical_filter_size_2 = self.vertical_filter_size_2
        min_line_length = self.min_line_length
        max_line_gap = self.max_line_gap
        long_line_factor = self.long_line_factor
        epsilon_v_1 = self.epsilon_v_1
        epsilon_h_1 = self.epsilon_h_1
        epsilon_v_2 = self.epsilon_v_2
        epsilon_h_2 = self.epsilon_h_2
        min_samples_1 = self.min_samples_1
        min_samples_2 = self.min_samples_2

        # ------ correct rotation of image, preprocess ------
        # print('step1')

        # set to float so that the following normalization works
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

        # set 25% of the lightest values to an equal value
        # to flatten out paper lightness variations
        # most of the paper is white. On the best case picture,
        # a value of 90% still works fine.
        # b = sorted(grey.flatten())
        # max_value = b[int(len(b)*0.75)]
        # grey[grey > max_value] = max_value

        # set levels between 0 and 255
        grey -= grey.min()
        # if invalid, return no handwriting detected
        if grey.max() == 0:
            return img
        grey = grey * 255 / grey.max()
        grey = grey.astype(np.dtype('uint8'))

        # transform the image into binary for hough lines detection
        # because it basically consists of edges only, no need to run
        # edge detection again before the hough line detection
        # bw = grey.copy()
        # mask = grey > bw_threshold
        # bw[mask == True] = 0
        # bw[mask == False] = 1
        bw = cv2.Canny(grey, 50, 150, apertureSize=3)
        # show(bw)
        # print('-', bw.sum())



        # ------ find vertical lines and median line height ------
        # print('step2')


        # detect vertical edges, handwritten text will (hopefully)
        # have a less dense detection rate
        vertical_lines = cv2.HoughLinesP(bw, 1, np.pi, 25, min_line_length, max_line_gap)

        # if invalid, return no handwriting detected
        if vertical_lines is None or len(vertical_lines) == 0:
            return img

        # First, find out the median, which will select
        # some height in machine written letters.
        line_lenghts = np.zeros(len(vertical_lines))
        for i, line in enumerate(vertical_lines):
            # the lines are vertical, so the lengths is simply |y1 - y2|
            # example for line: [[271 809 271 799]]
            line_lenghts[i] = abs(line[0][1] - line[0][3])
        line_med = int(np.median(line_lenghts))

        # check if line_med is a unrealistic value to prevent memory errors
        # and save computational time by stopping computations that lead
        # to nothing anyway
        if line_med > img.shape[0] >> 5 or line_med > img.shape[1] >> 5:
            # print('line_med faulty')
            return img



        # ------ remove long vertical and horizontal lines ------
        # print('step3')

        # detect horizontal lines only, by transposing and set angle to 180deg again 
        long_lines_image = np.zeros(grey.shape) # <- that is the empty image
        horizontal_lines = cv2.HoughLinesP(bw.T, 1, np.pi, 15, min_line_length, max_line_gap)
        if not horizontal_lines is None:
            for x in range(0,  len(horizontal_lines)):
                # was working on the transposed image (which rotates it by 90deg),
                # so it is y, x, ... not x, y, ....
                for y1, x1, y2, x2 in horizontal_lines[x]:
                    if abs(x2 - x1) > line_med * long_line_factor:
                        cv2.line(long_lines_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # now also go over the vertical lines again
        for x in range(0,  len(vertical_lines)):
            for x1, y1, x2, y2 in vertical_lines[x]:
                if abs(y2 - y1) > line_med * long_line_factor:
                    cv2.line(long_lines_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # remove long lines from the image
        grey[long_lines_image > 128] = grey.max()
        bw[long_lines_image > 128] = bw.max()


        # ------ remove vertical lines that are noise ------

        # 'DBSCAN' step 1 --------
        # basically just look for one single neighbor,
        # which will remove noise basically.
        # basically looks for straight lines.

        # find out which lines have a 'support' to the right
        # because machine written text is super straight (basically)

        # only use the lower of the two points per line (that is the first point)
        # vertical lines is an array like [[[1,2,3,4]], [[5,6,7,8]]]
        # transform to [[1,2], [5,6]]
        bottom_points = vertical_lines[:,:,2:].reshape(vertical_lines.shape[0], 2)
        labels = self.rdbscan(bottom_points, 150, 10, min_samples=15)
        # labels = self.rdbscan(bottom_points, epsilon_v_1, epsilon_h_1, min_samples=min_samples_1)
        vertical_valid_lines = vertical_lines[labels != -1]
        vertical_invalid_lines = vertical_lines[labels == -1]

        if len(vertical_valid_lines) == 0:
            # print('no valid vertical lines found. n of invalid lines:', len(vertical_invalid_lines))
            return img

        if verbose:
            self.show_lines(img, vertical_valid_lines, vertical_invalid_lines)


        # 'DBSCAN' step 2 --------
        # cluster machine written text

        # hough gives coordinate pairs of [[x1, y1, x2, y2]]
        # transform that into [[x1, y1], [x2, y2]]
        points = np.array(vertical_valid_lines).flatten()
        points = points.reshape((points.size >> 1, 2)).astype(float)

        labels = self.rdbscan(points, epsilon_v_2, epsilon_h_2, min_samples=min_samples_2)

        # since lines were splitted into start and entpoints before,
        # figure out which lines have both their start and endpoint in a DBSCAN label.
        # now [1, -1, 2, 1, -1, -1] is transformed to: [[1, -1], [2, 1], [-1, -1]]
        a = (len(labels) >> 1, 2)
        b = labels.reshape(a)
        # figure out which values are clustered (!= -1): [[True, False], ...]
        c = b != -1
        # then sum them to: [1, 0, 2]
        d = c.sum(axis=1)
        # then transform to [True, True, False]
        mask_2 = d > 0
        
        # now remove those points from vertical_valid_lines that have
        # the label -1, which means no label, which hopefully corresponds
        # to a bunch of handwritten stuff
        vertical_invalid_lines = vertical_valid_lines[mask_2 == False]
        vertical_valid_lines = vertical_valid_lines[mask_2]

        if len(vertical_valid_lines) == 0:
            # print('no valid vertical lines found. n of invalid lines:', len(vertical_invalid_lines))
            return img

        if verbose:
            self.show_lines(img, vertical_valid_lines, vertical_invalid_lines)


        # ------ find density of vertical lines ------

        # resize to improve performance. for the "density" calculation,
        # no high resolution is required, really.
        # this parameter can be changed even after the parameters are optimized,
        # won't change the result too much.
        bits = 3
        filter_size_multiplicator_1

        grey_small = cv2.resize(grey, (grey.shape[1]>>bits, grey.shape[0]>>bits))

        # create an empty image
        vertical_lines_image = np.zeros(grey_small.shape)

        # fill in the valid vertical lines
        for x in range(0,  len(vertical_valid_lines)):
            for x1, y1, x2, y2 in vertical_valid_lines[x]:
                cv2.line(vertical_lines_image, (x1>>bits, y1>>bits), (x2>>bits, y2>>bits), (255, 255, 255), 2)

        img_blur = self.blur(vertical_lines_image, int(line_med * (filter_size_multiplicator_1)) >> bits, vertical_filter_size_1)
        img_blur /= img_blur.max()

        # show(vertical_lines_image)

        # if invalid, return no handwriting detected
        if img_blur.max() <= 0:
            return img

        img_blur /= img_blur.max()
        mask = img_blur > density_threshold_1

        # remove machine_written from the grayscale image
        grey_small[mask] = grey_small.max()


        # ------ detect handwriting in leftover image, remove noise ------
        # print('step4')

        final = img

        # Basically cluster the leftovers. in grey, elements
        # are removed in the previous code.
        # Blur, which will assign dark values to cramped areas,
        # then threshold.
        img_blur = self.blur(grey_small, int(line_med * (filter_size_multiplicator_2)) >> bits, vertical_filter_size_2)
        img_blur = cv2.resize(img_blur, (grey.shape[1], grey.shape[0]))
        final[:,:,0][img_blur > (density_threshold_2)] = final[:,:,0].max()
        final[:,:,1][img_blur > (density_threshold_2)] = final[:,:,1].max()
        final[:,:,2][img_blur > (density_threshold_2)] = final[:,:,2].max()
        final[img_blur < (density_threshold_2)] >> 1 # make everything dark that is not handwriting

        if verbose:
            show(img_blur)

        # show(final)

        if repititions > 1:
            return self.detect_handwriting(final, verbose, repititions-1)
        else:
            return final

        # ------------- links -------------

        # hough
        # - https://stackoverflow.com/questions/33541551/hough-lines-in-opencv-python#33542934
        # - https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html

        # filter
        # - https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html

        # other stuff
        # - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html#display-image
        # - https://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html
        # - https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html#py-table-of-content-feature2d
