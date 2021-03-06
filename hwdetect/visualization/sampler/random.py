"""Module for sampling an image.

This module provides tools for selecting samples from an image and making predictions on these samples. The resulting
 predictions are used by the heat map module to create heat maps.
"""

__author__ = "Dennis Kraft"
__version__ = "1.0"

import numpy as np
from .sampler import Sampler, logger
from hwdetect.utils import show


class Random(Sampler):
    """An object for sampling predictions at randomly drawn coordinates of an image."""

    def __init__(self, sample_stepsize=1000):
        """Create a sampler that draws samples at random.

        no need to supply width and height of the samples, as this
        value is provided by the model.

        Parameters
        ----------
        sample_stepsize : int, optional
            The frequency with which samples are taken from the image. For instance, a sample_stepsize of 1000
            corresponds to 1 sample per 1000 pixels of the image.
        """
        
        self.sample_stepsize = sample_stepsize

    def sample(self, image, predictor, label_aggregator, original):
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

        # get size of image
        height = image.shape[0]
        width = image.shape[1]

        # pad image
        sample_size = predictor.get_image_size()
        border = sample_size // 2
        padded_preprocessed = np.pad(image, ((border, border + 1), (border, border + 1), (0, 0)), 'edge')
        padded_original = np.pad(original, ((border, border + 1), (border, border + 1), (0, 0)), 'edge')

        # draw random sample coordinates without replacement
        coordinates = np.random.choice(height * width, (height * width) // self.sample_stepsize, replace=False)

        # make predictions
        progress = 0
        step_size = len(coordinates) // 10
        predictions = {}
        skipped_count = 0
        for i in range(0, len(coordinates)):
            # print progress
            if step_size > 0 and i % step_size == 0:
                logger.info("{}% of predictions complete".format(progress))
                progress += 10

            # make prediction and add to the sample dictionary
            y = coordinates[i] // width
            x = coordinates[i] % width
            chunk = [padded_preprocessed[y:y + sample_size, x:x + sample_size]]
            if np.var(chunk[0]) > 300:
                chunk = [padded_original[y:y + sample_size, x:x + sample_size]]
                predictions[(y, x)] = label_aggregator(predictor.predict(chunk)[0])
            else:
                skipped_count += 1
                predictions[(y, x)] = 0

        logger.info('skipped {} of {} chunks'.format(skipped_count, len(predictions)))

        return predictions
