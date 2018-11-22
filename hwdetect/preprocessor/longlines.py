import cv2
import numpy as np
from hwdetect.preprocessor.preprocessor import Preprocessor
# from hwdetect.utils import show

class Longlines(Preprocessor):
    def __init__(self, long_line_threshold=100, passes=10):
        """tries to remove long lines from the image,
        such as underlines"""

        self.long_line_threshold = long_line_threshold
        self.passes = passes

    def longlines(self, img):

        img = img.copy()

        # empty mask for now, which is going to be
        # filled with True values where long lines are
        mask = np.zeros(img.shape, np.uint8) 

        bw = np.zeros(img.shape, img.dtype)
        bw[img < 200] = 1
        # bw = cv2.Canny(img, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(bw, 1, np.pi/180, 100)
        
        # now go over the lines and fill in True values into the mask
        for i in range(0,  len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                if np.linalg.norm((x1 - x2, y1 - y2)) > long_line_threshold: 
                    cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 3)

        img[mask == 255] = img.max()

        return img
        