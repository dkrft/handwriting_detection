import cv2
import numpy as np
from scipy import ndimage
from hwdetect.utils import show
from hwdetect.preprocessor.preprocessor import Preprocessor

# also see:
# http://felix.abecassis.me/2011/09/opencv-detect-skew-angle/
# but using median instead of mean, because counter-clockwise rotated
# pictures could not be corrected otherwise.

class Deskew(Preprocessor):

    def __init__(self, keep, keep_dimensions=False):
        """Corrects the angle of incoming images.
        If keep_dimensions is True, will crop the
        image to its original width and height, potentially
        cropping some text away. Default: False"""
        self.keep_dimensions = keep_dimensions

    def preprocess(self, img):
        """
        if angle is None, will try to automatically detect the angle
        using the median of the angles of hough lines.

        Parameters
        ----------
        img : array of shape (x, y, 3) or (x, y)
            grayscale or RGB image. width and height
            can be arbitrary

        Returns
        -------
            the rotated image

        """

        keep_dimensions = self.keep_dimensions

        width = img.shape[1]

        # edges = cv2.Canny(img.astype(np.uint8), 50, 150, apertureSize = 3)
        edges = np.ones(img.shape, img.dtype)
        # make white px in the original image black
        edges[img > 220] = 0
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, width/4, 20)
        # example for lines array before reshape: [[[259 568 288 572]], [[...]], ...]
        lines = lines.reshape((len(lines), 4))

        # transpose, because the result of vectors_centered will be [substractions1, substractions2]
        # and i want a list of 2-tuples [(sub1, sub2), (sub1, sub2), ...] for each line
        # vectors_centered = np.array([lines[:,3] - lines[:,1], lines[:,2] - lines[:,0]]).T
        # but then, arctan2 actually does not want it transposed kek

        vectors_centered = np.array([lines[:,3] - lines[:,1], lines[:,2] - lines[:,0]])
        angles = np.arctan2(vectors_centered[0], vectors_centered[1])

        angle = np.median(angles)

        # rotate
        # rotation angle is in degrees (that is, the parameter of rotate)
        rotated = ndimage.rotate(img, angle * 360 / np.pi / 2, cval=img.max())

        # crop, such that the output image is the same as the input
        if keep_dimensions:
            a = int((rotated.shape[0] - img.shape[0])/2)
            b = int((rotated.shape[1] - img.shape[1])/2)
            rotated = rotated[a:-a-1, b:-b-1]

        return rotated
        