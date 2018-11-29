#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Tobias B <github.com/sezanzeb>"
__version__ = "1.0"


import hwdetect
import cv2
import time
from hwdetect.utils import show, get_path

from sklearn.neighbors import KNeighborsRegressor
from hwdetect.visualization.interpolation import NearestNeighbour
from hwdetect.preprocessor import Bandpass
from hwdetect.neural_network import Predictor

import matplotlib.pyplot as plt

import logging

# enable INFO on all loggers in the hwdetect packages
logger = logging.getLogger('hwdetect')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

img = cv2.imread(get_path('examples/example_data/easy1.jpg'))

"""filtered = Bandpass().filter(img)
show(filtered)
quit()"""


# heat_map
start = time.time()
heat_map = hwdetect.visualization.create_heat_map(img,
            preprocessors=[hwdetect.preprocessor.Scale(), hwdetect.preprocessor.Bandpass()],
            sampler=hwdetect.visualization.sampler.Stride(),
            predictor=hwdetect.neural_network.Predictor(gpu=1),
            interpolator=KNeighborsRegressor(),
            postprocessors=[],
            heat_map_scale=20)
print(round(time.time() - start, 3), 'Seconds')
hwdetect.visualization.plot_heat_map(img, heat_map)


# bounding boxes
boxes = hwdetect.visualization.bounded_image(img, heat_map, perc_thresh=0.9)
show(boxes)


