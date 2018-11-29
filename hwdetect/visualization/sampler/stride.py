#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Tobias B <github.com/sezanzeb>"
__version__ = "1.0"


import numpy as np
from .sampler import Sampler, logger
from hwdetect.utils import show
from random import uniform
import cv2

class Stride(Sampler):

    def __init__(self, stride=None, y_random=0):
        """Samples chunks out of the image using a sliding window.
        If the chunk contains dark pixels in the top but not at the bottom,
        slides the chunk down in order to center the text so that it can
        be properly classified. Does the same for text on the bottom as well.

        Parameters
        ----------
        stride : int, optional
            by how much to slide the window in x and y direction each step.
            If None, will take 1/3 of the predictors sample size.
        y_random : float, optional
            percent of the stride to randomly vary in y
            direction between - and + y_random. Default: 0.1
        """
        
        self.stride = stride
        self.y_random = y_random


    def sample(self, image, predictor, label_aggregator, original=None):
        """Draw samples from the specified image and predict their labels.

        Parameters
        ----------
        image : np.array
            The image from which the samples are drawn
        predictor : neural_network.predictor.Predictor
            The predictor object that is used for predicting the labels of a sample.
        label_aggregator : function from list of floats to float
            The function that is used for combining the labels predicted by the predictor into a single label.

        Returns
        -------
        dict mapping tuple of int to float
            A dictionary containing the sampled predictions. The keys of the dictionary correspond to the positions of
            the samples while values of the dictionary correspond to the predicted label. The positions is encoded as a
            tuple of integers such that the first integer denotes the y-coordinate and the second integer denotes the
            x-coordinate.
        """

        if original is None:
            original = image

        # get size of image
        height = image.shape[0]
        width = image.shape[1]
        stride = self.stride
        sample_size = predictor.get_image_size()

        if stride is None:
            stride = sample_size//3
            
        # pad image
        """# shorthands:
        c = sample_size # chunk size
        w = width
        h = height
        s = self.stride

        # explanation of the formula:
        # 1.) the stride can be executed w//s times until the left edge of the chunk would be !outside! of the image
        # when that left edge index is the same as the last index of the image + 1, that case will be covered later (A)
        # 2.) no see how many px that actually are by multiplying with the stride
        # 3.) see what the difference to w is, that means, how many px from the right is the left edge of the chunk
        # 4.) look at the difference to c, which is how many px of the chunk would be missing in the image. that's the padding
        # 5.) now for (A). It might be that this value is equal to c, so modulo with c to get 0 which means it fits perfectly
        border_h = (c - (w - (w//s)*s)) % c
        border_v = (c - (h - (h//s)*s)) % c"""

        # well, or you just add the chunk size onto it which would prevent stepping over it
        # for good with 100% more easy to debug code tadaa with a small extra memory overhead

        padded_img = np.pad(image, ((0, sample_size), (0, sample_size), (0, 0)), 'edge')
        padded_original = np.pad(original, ((0, sample_size), (0, sample_size), (0, 0)), 'edge')


        X = range(0, width-sample_size, stride)
        Y = range(0, height-sample_size, stride)

        total_num_predictions = len(X) * len(Y)

        skipped_count = 0

        # make predictions
        progress = 0
        predictions = {}

        for y_grid in Y:
            for x in X:

                # gonna overwrite y with some adjustments
                # don't keep them from the previous iteration
                y = y_grid

                # print approx every 10 percent
                if progress % np.ceil(total_num_predictions/(100/10)) == 0:
                    logger.info("{}% of predictions complete".format(int(progress/total_num_predictions*100)))

                y += int(sample_size * uniform(-self.y_random, self.y_random))
                y = max(0, min(height, y))

                #[None, :] adds another axis to the left of the shape: (1, ...)
                # -> wraps the whole thing into square brackets
                chunk = padded_img[y:y + sample_size, x:x + sample_size][None, :]

                # variances of chunks when accepting based on mean threshold of 250
                #                  mean              max                min
                # accepted chunks: 1774.933370429126 4437.332382468917  673.7359007991221
                # declined chunks: 35.23593238683097 1015.4963759451304 0.0

                if np.var(chunk[0]) > 300:
                # if np.mean(chunk[0]) < 254:
                
                    # the ordering of the array doesn't make a difference, as can be seen here:
                    # which means it looks as the points value in one dimension, and doesn't treat
                    # the index as axis.
                    #
                    # >>> np.var([np.random.randint(0,2) for _ in range(1000)])
                    # 0.249984
                    # >>> np.var([0]*500 + [1]*500)
                    # 0.25

                    # look at where the dark values in that cunk are, make histogramm over rows sum lightness
                    # move to center of chunk towards the dark values.
                    res = 5 # historgram resolution
                    hist = cv2.resize(chunk[0].min(axis=2), (1, res), interpolation=cv2.INTER_AREA)
                    darkest = np.argmin(hist)
                    # darkest has to be either 0 or 4 for res=5, so res-darkest-1
                    # would be the opposite side. If there on the opposite side
                    # is the lightest value, move towards the dark value. 
                    if hist[res-darkest-1] == hist.max():
                        # step = sample_size/(res-1)
                        step = stride//2
                        if darkest == 0:
                            y -= step
                        if darkest == res-1:
                            y += step

                    y = max(0, min(height, int(y)))

                    chunk = padded_original[y:y + sample_size, x:x + sample_size][None, :]

                    prediction = predictor.predict(chunk)[0]
                    predictions[(y + sample_size//2, x + sample_size//2)] = label_aggregator(prediction)
                else:
                    skipped_count += 1
                    predictions[(y + sample_size//2, x + sample_size//2)] = 0

                progress += 1

        logger.info('skipped {} of {} chunks'.format(skipped_count, total_num_predictions))

        # visualize sampled points
        """original = original//2
        for point in predictions:
            val = predictions[point]
            y, x = point
            cv2.circle(original, (x, y), 10, (128, 255-int(val*255), int(val*255)), -10)
        show(original)"""

        return predictions
