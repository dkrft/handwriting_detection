#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Tobias B <github.com/sezanzeb>"
__version__ = "1.0"


from .postprocessor import Postprocessor

class Threshold(Postprocessor):

    def __init__(self, upper=1, lower=0.75):
        self.upper = upper
        self.lower = lower

    def filter(self, heatmap):
        """
        thresholds the heatmap in order to cut away irrelevant
        values

        """

        upper = self.upper
        lower = self.lower

        # set to float for more flexible computation
        ret = heatmap.copy()
        ret[heatmap > upper] = upper
        ret[heatmap < lower] = lower

        # normalize to between 0 and 255
        ret -= lower
        ret *= 255
        ret /= (upper-lower)

        return ret
        