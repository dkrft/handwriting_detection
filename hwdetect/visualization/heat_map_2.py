"""Module for visualizing predictions as heat maps.

This module provides tools for creating and visualizing heat maps. The heat map of a given image is created  by taking
samples from the image, predicting the labels of the samples and assembling the predictions in the heat map.

Example
-------

Train a neuronal network to identify the bright areas of a chessboard and visualize its predictions in a heat map.

    >>> import numpy as np
    >>> import os
    >>> from data.training_data import TrainingData
    >>> from neural_network import model
    >>> from neural_network.predictor import Predictor
    >>> from visualization import heat_map
    >>> from visualization.pixel_interpolation import Interpolator
    >>>
    >>> # create three 10 by 10 pixel images with 3 color channels and store them in a data set for training
    >>> black_image = np.full((10, 10, 3), 0)
    >>> white_image = np.full((10, 10, 3), 255)
    >>> grey_image = np.full((10, 10, 3), 127)
    >>> training_data = TrainingData([black_image, white_image], [[0], [1]], [grey_image], [[1]])
    >>>
    >>> # setup the path to the directory where the model of the neural network is saved
    >>> cwd = os.getcwd()
    >>> model_directory = os.path.join(cwd, "brightness_detector")
    >>>
    >>> # create and train the model
    >>> model.create(model_directory, 10, 3, 1)
    >>> model.train(model_directory, training_data, max_iterations=15, save_frequency=2, max_save=5)
    >>>
    >>> # create an image of a chess board
    >>> black_square = np.full((100, 100, 3), 0)
    >>> white_square = np.full((100, 100, 3), 255)
    >>> column_1 = np.concatenate((white_square, black_square, white_square, black_square,
    >>>                            white_square, black_square, white_square, black_square), axis=0)
    >>> column_2 = np.concatenate((black_square, white_square, black_square, white_square,
    >>>                            black_square, white_square, black_square, white_square), axis=0)
    >>> chess_board = np.concatenate((column_1, column_2, column_1, column_2,
    >>>                               column_1, column_2, column_1, column_2), axis=1)
    >>>
    >>> # create a heat map of the bright areas of the chessboard image using the most recent training iteration of the
    >>> # brightness detector model
    >>> chess_board_hm = heat_map.create_heat_map(chess_board, Predictor(model_directory), sample_frequency=50)
    >>>
    >>> # show the heat map
    >>> heat_map.show_heat_map(chess_board, chess_board_hm)
"""

__author__ = "Dennis Kraft"
__version__ = "1.0"

import numpy as np
import cv2
from matplotlib import pyplot as plt
from .interpolation import NearestNeighbour
from .sampler import RandomGrid
from scipy.spatial.distance import cdist
from hwdetect.utils import show
from sklearn.neighbors import KNeighborsRegressor
import time


def create_heat_map_2(image, predictor,
                    label_aggregator=lambda labels: labels[0],
                    sampler=RandomGrid(),
                    preprocessors=[],
                    interpolator=NearestNeighbour(),
                    heat_map_scale=10):
    """Create a heat map of an image based on the predictions made by the specified neural network model.

    Parameters
    ----------
    image : np.array
        The image for which the heat map is created
    predictor : neural_network.predictor.Predictor
        The predictor object that is used for predicting the heat map.
    label_aggregator : function from list of floats to float, optional
        The function that is used for combining the labels predicted by the predictor into a single label. By default,
        only the first label is used.
    sampler : class implementing visualization.pixel_interpolation.RandomSampler
        The sampler that is used for drawing samples from the image and predicting their labels.
    preprocessors : list of functions from np.array to np.array
        A list of the image processing functions that are applied to the original image before starting the sampling and
        prediction phase. The preprocessors are applied in the order of the list.
    interpolator : class implementing visualization.pixel_interpolation.Interpolator
        The interpolator that is used to infer the pixels of the heat map based on the predictions made in the sampling
        and prediction phase.
    heat_map_scale : int
        The resolution of the heat map. For instance, a resolution of 5 implies that squares of 5 by 5 pixels in the
        original image are condensed into 1 pixel in the heat map.

    Returns
    -------
    np.array
        A two dimensional array representing the heat map. Note that the height and width of the array matches the
        height and width of the original image scaled down by the heat map resolution parameter.
    """

    # set up the heat map
    height = image.shape[0]
    width = image.shape[1]
    heat_map = np.zeros((height // heat_map_scale, width // heat_map_scale))

    # pre-process image
    for preprocessor in preprocessors:
        image = preprocessor.preprocess(image)
    print('pre-processing complete')

    # make predictions
    predictions = sampler.sample(image, predictor, label_aggregator)

    X_pred = [k for k in predictions]
    Y_pred = [predictions[k] for k in predictions]
    interpolator.fit(X_pred, Y_pred) 

    # create grid for all the coordinates in the heatmap
    # [0] is height, [1] is width
    Y, X = np.mgrid[:heat_map.shape[0], :heat_map.shape[1]]
    coords = [a for a in zip(Y.flatten(), X.flatten())]
    coords_scaled = np.array([a for a in zip(Y.flatten(), X.flatten())])*heat_map_scale

    print('interpolating...')
    values = interpolator.predict(coords_scaled)

    heat_map = values.reshape(heat_map.shape)

    print('done')

    return heat_map


def plot_heat_map(image, heat_map):
    """Overlay an image with the specified heat map and plot the result.

    Parameters
    ----------
    image : np.array
        The image that is plotted.
    heat_map : np.array
        The heat map put on top of the image
    """
    height, width, _ = image.shape
    plt.imshow(image)
    plt.imshow(cv2.resize(heat_map, (width, height), interpolation=cv2.INTER_NEAREST),
               cmap=plt.cm.viridis,
               alpha=.6,
               interpolation='bilinear')
    plt.xticks([])
    plt.yticks([])
    plt.show()
