"""Module for sampling an image.

This module provides tools for selecting samples from an image and making predictions on these samples. The resulting
 predictions are used by the heat map module to create heat maps.
"""

__author__ = "Dennis Kraft"
__version__ = "1.0"


class Sampler:
    """A template for a sampling objects."""

    def __init__(self):
        """Create a sampler."""
        pass

    def sample(self, image, predictor):
        """Draw samples from the specified image and predict their labels.

        Parameters
        ----------
        image : np.array
            The image from which the samples are drawn
        predictor : neural_network.predictor.Predictor
            The predictor object that is used for predicting the labels of a sample.

        Returns
        -------
        dict mapping tuple of int to float
            An empty dictionary containing the sampled predictions. Note that the purpose of this implementation is to
            serve as a template. In particular, it does not provide a sensible collection of predictions.
        """
        return {}
