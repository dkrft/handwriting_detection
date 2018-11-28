#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Tobias B <github.com/sezanzeb>"
__version__ = "1.0"


class Preprocessor():
    """preprocessor base class to inherit from"""
    
    def filter(self, img):
        raise NotImplementedError()
        return img