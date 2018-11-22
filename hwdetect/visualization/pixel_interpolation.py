"""Module for interpolating the pixels of an image based on a limited subset of sampled pixels.

This module provides tools for interpolating the pixels of an image based on a limited set of sampled  pixels. These
interpolation techniques are used by the heat map module to speed up the prediction process and reduce noise.
"""

__author__ = "Dennis Kraft"
__version__ = "1.0"

import numpy as np
import scipy as sp


class Interpolator:
    """A template for interpolator objects."""

    def __init__(self):
        """Create an interpolator."""
        pass

    def fit(self, samples):
        """Load the samples that are used for interpolation.

        Parameters
        ----------
        samples : dict mapping tuple of int to float
            A dictionary containing the samples that are used for interpolation. The keys of the dictionary correspond
            to the positions of the samples while values of the dictionary correspond to the color of the pixel. The
            positions must be encoded as a tuple of integers where the first integer denotes the y-coordinate and the
            second integer denotes the x-coordinate.
        """
        pass

    def interpolate(self, y, x):
        """Interpolate the color of a pixel at the specified coordinates.

        Parameters
        ----------
        y : int
            Y-coordinate of the pixel that is interpolated.
        x: int
            X-coordinate of the pixel that is interpolated.

        Returns
        -------
        float
            An interpolation of the color at the specified coordinates based on a randomly drawn float. Note that the
            purpose of this implementation is to serve as a template for sub classes. In particular, it does not provide
            a sensible interpolation of the color of a pixel.
        """
        return np.random.rand()


class NearestNeighbourInterpolator(Interpolator):
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

    def fit(self, samples):
        """Load the samples that are used for interpolation.

        Parameters
        ----------
        samples : dict mapping tuple of int to float
            A dictionary containing the samples that are used for interpolation. The keys of the dictionary correspond
            to the positions of the samples while values of the dictionary correspond to the color of the pixel. The
            positions must be encoded as a tuple of integers where the first integer denotes the y-coordinate and the
            second integer denotes the x-coordinate.
        """

        # copy samples and create kdtree based on the sample coordinates
        self.samples = {key: samples[key] for key in samples.keys()}
        self.kdtree = sp.spatial.KDTree([key for key in samples.keys()])

    def interpolate(self, y, x):
        """Interpolate the color of a pixel at the specified coordinates.

        Parameters
        ----------
        y : int
            Y-coordinate of the pixel that is interpolated.
        x: int
            X-coordinate of the pixel that is interpolated.

        Returns
        -------
        float
            An interpolation of the color at the specified coordinates based on the color of the closest samples.
        """

        # get he indices of the samples closest to the specified position
        _, neighbour_indices = self.kdtree.query([(y, x)], k=min(len(self.samples), self.neighbours_to_consider))

        # retrieve the position of the selected samples from their indices
        neighbour_positions = [self.kdtree.data[i] for i in neighbour_indices[0]]

        # look up the values of the samples based on their position
        neighbour_values = [self.samples[(neighbour_position[0], neighbour_position[1])]
                            for neighbour_position in neighbour_positions]

        # return the aggregation of the samples
        return self.aggregator(neighbour_values)
