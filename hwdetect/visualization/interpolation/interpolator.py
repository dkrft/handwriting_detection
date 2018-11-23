"""Module for interpolating the pixels of an image based on a limited subset of sampled pixels.

This module provides tools for interpolating the pixels of an image based on a limited set of sampled  pixels. These
interpolation techniques are used by the heat map module to speed up the prediction process and reduce noise.
"""

__author__ = "Dennis Kraft"
__version__ = "1.0"

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