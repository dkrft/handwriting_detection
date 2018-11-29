"""Module for interpolating the pixels of an image based on a limited subset of sampled pixels.

This module provides tools for interpolating the pixels of an image based on a limited set of sampled  pixels. These
interpolation techniques are used by the heat map module to speed up the prediction process and reduce noise.
"""

__author__ = "Dennis Kraft"
__version__ = "1.0"

import numpy as np
import scipy as sp
from .interpolator import Interpolator, logger


class NearestNeighbour(Interpolator):
    """An object to interpolate pixels based to the values of the sampled pixels closest to them."""

    def __init__(self,
                 num_of_neighbours=12,
                 aggregator=lambda neighbours: np.quantile(neighbours, 0.33, interpolation='nearest')):
        """Create a nearest neighbour interpolator.

        Parameters
        ----------
        num_of_neighbours : int, optional
            The number of neighbours that are considered for the interpolation
        aggregator: function from list of floats to float
            The function that is used to interpolate the color of a pixel based on the color of its nearest neighbours.
        """

        # set number of neighbours and aggregator as specified
        self.neighbours_to_consider = num_of_neighbours
        self.aggregator = aggregator

        # create place holders for storing the samples
        self.samples = None
        self.kdtree = None

    def fit(self, samples, values=None):
        """Load the samples that are used for interpolation.

        Parameters
        ----------
        samples : dict mapping tuple of int to float
            A dictionary containing the samples that are used for interpolation. The keys of the dictionary correspond
            to the positions of the samples while values of the dictionary correspond to the color of the pixel. The
            positions must be encoded as a tuple of integers where the first integer denotes the y-coordinate and the
            second integer denotes the x-coordinate.

            or if value is not None: an array of coordinates (as with sklearn classes)

        values : array
            if None, will treat samples as dict of {coord: value}.
            if it is an array, will treat samples as array of coordinates.
        """

        # copy samples and create kdtree based on the sample coordinates
        if not values is None:
            self.samples = {samples[i]: values[i] for i in range(len(samples))}
        else:
            self.samples = {key: samples[key] for key in samples.keys()}
            
        self.kdtree = sp.spatial.KDTree([key for key in self.samples.keys()])

    def predict(self, X):
        """Interpolate the color of a pixel at the specified coordinates.

        Parameters
        ----------
        X : array
            array of (y, x) tuples, y and x being integer coordinates
            of the pixel that should be interpolated.
            
        Returns
        -------
        float
            An interpolation of the color at the specified coordinates based on the color of the closest samples.
        """

        ret = np.zeros(len(X))

        progress = 1
        for i, coord in enumerate(X):

            if i == int(len(X)/10)*progress:
                logger.info("{}% of interpolation complete".format((progress-1)*10))
                progress += 1

            # unpack tuple
            y, x = coord

            # get he indices of the samples closest to the specified position
            _, neighbour_indices = self.kdtree.query([(y, x)], k=min(len(self.samples), self.neighbours_to_consider))

            # happens when num_of_neighbours is 1:
            if len(neighbour_indices.shape) == 1:
                neighbour_indices = neighbour_indices[:,None]

            # retrieve the position of the selected samples from their indices
            neighbour_positions = [self.kdtree.data[i] for i in neighbour_indices[0]]

            # look up the values of the samples based on their position
            neighbour_values = [self.samples[(neighbour_position[0], neighbour_position[1])]
                                for neighbour_position in neighbour_positions]

            # return the aggregation of the samples
            ret[i] = self.aggregator(neighbour_values)

        return ret
