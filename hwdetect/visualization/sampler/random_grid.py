"""Module for sampling an image.

This module provides tools for selecting samples from an image and making predictions on these samples. The resulting
 predictions are used by the heat map module to create heat maps.
"""

__author__ = "Dennis Kraft"
__version__ = "1.0"

import numpy as np
from .sampler import Sampler
from hwdetect.utils import show

class RandomGrid(Sampler):
    """An object for sampling predictions at randomly drawn coordinates of an image but with a fixed number of samples
    per cell of an evenly sapced grid.
    """

    def __init__(self, grid_frequency=25, sample_frequency=1000):
        """Create a sampler that draws samples evenly at random within a predefined grid.

        no need to supply width and height of the samples, as this
        value is provided by the model.
        
        Parameters
        ----------
        grid_frequency : int, optional
            The frequency with which grid_lines are drawn. For instance, a grid frequency of 25 corresponds to one grid
            line for every 25 pixels.
        sample_frequency : int, optional
            The frequency with which samples are taken from the image. For instance, a sample frequency of 1000
            corresponds to 1 sample per 1000 pixels of the image.
        """

        self.grid_frequency = grid_frequency
        self.sample_frequency = sample_frequency

    def sample(self, image, predictor, label_aggregator, original=None):
        """Draw samples from the specified image and predict their labels.

        Parameters
        ----------
        image : np.ndarray
            The image from which the samples are drawn
        predictor : neural_network.predictor.Predictor
            The predictor object that is used for predicting the labels of a sample.
        label_aggregator : function from list of floats to float
            The function that is used for combining the labels predicted by the predictor into a single label. By
            default, only the first label is used.
        original : np.ndarray
            The original image. Default: None will use the preprocessed image

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

        # get size of image and grid cells
        height = image.shape[0]
        width = image.shape[1]
        cell_size = self.grid_frequency
        cell_area = cell_size * cell_size

        # pad image
        sample_size = predictor.get_image_size()
        border = sample_size // 2
        padded_preprocessed = np.pad(image, ((border, border + 1), (border, border + 1), (0, 0)), 'edge')
        padded_original = np.pad(original, ((border, border + 1), (border, border + 1), (0, 0)), 'edge')

        # make predictions
        progress = 0
        step_size = ((width // cell_size) * (height // cell_size)) // 10
        predictions = {}
        I = range(0, height // cell_size)
        J = range(0, width // cell_size)
        skipped_count = 0
        for i in I:
            for j in J:
                # print progress
                if step_size > 0 and (j + (i * (width // cell_size))) % step_size == 0:
                    print("{}% of predictions complete".format(progress))
                    progress += 10

                # draw random sample coordinates without replacement
                coordinates = np.random.choice(cell_area, cell_area // self.sample_frequency + 1, replace=False)

                # make prediction and add to the sample dictionary
                for coordinate in coordinates:
                    y = (cell_size * i) + (coordinate // cell_size)
                    x = (cell_size * j) + (coordinate % cell_size)
                    chunk = [padded_preprocessed[y:y + sample_size, x:x + sample_size]]

                    if np.var(chunk[0]) > 200:

                        chunk = [padded_original[y:y + sample_size, x:x + sample_size]]
                        pred = predictor.predict(chunk)[0]
                        predictions[(y, x)] = label_aggregator(pred)
                        # print(pred)
                        # show(chunk[0])
                    else:
                        skipped_count += 1
                        predictions[(y, x)] = 0

        total_num_predictions = len(I) * len(J)
        print('skipped {} of {} chunks'.format(skipped_count, total_num_predictions))
        
        return predictions
