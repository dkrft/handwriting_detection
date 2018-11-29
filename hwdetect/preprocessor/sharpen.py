#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Tobias B <github.com/sezanzeb>"
__version__ = "1.0"


import cv2
import numpy as np
from scipy import ndimage
from hwdetect.utils import show
from hwdetect.preprocessor.preprocessor import Preprocessor

class Sharpen(Preprocessor):
    """
    returns the sharpened version of img

    constructs a function basically that can be called

    usage: sharpen(0.5)()
    """

    def __init__(self, intensity=0.75):
        self.intensity = intensity

    def filter(self, img):
        intensity = self.intensity
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        ret = (cv2.filter2D(img, -1, kernel) * intensity) + (img * (1-intensity))

        return ret.astype(np.uint8)
        