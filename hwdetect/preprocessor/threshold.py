#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Tobias B <github.com/sezanzeb>"
__version__ = "1.0"


from .preprocessor import Preprocessor

class Threshold(Preprocessor):

    def __init__(self, upper=220, lower=30):
        self.upper = upper
        self.lower = lower

    def filter(self, img):
        """
        removes noise ,e.g. from jpg compression or just random noise
        by thresholding the image

        """

        upper = self.upper
        lower = self.lower

        # set to float for more flexible computation
        ret = img.copy().astype(float)
        ret[img > upper] = upper
        ret[img < lower] = lower

        # normalize to between 0 and 255
        ret -= lower
        ret *= 255
        ret /= (upper-lower)

        # set type back
        ret = ret.astype(img.dtype)

        return ret
        