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
from visualization.pixel_interpolation import NearestNeighbourInterpolator


def create_heat_map(image, predictor,
                    sample_frequency=1000,
                    label_aggregator=lambda labels: labels[0],
                    preprocessors=[],
                    interpolator=NearestNeighbourInterpolator(),
                    heat_map_resolution=10):
    """Create a heat map of an image based on the predictions made by the specified neural network model.

    Parameters
    ----------
    image : np.array
        The image for which the heat map is created
    predictor : neural_network.predictor.Predictor
        The predictor object that is used for predicting the heat map.
    sample_frequency : int, optional
        The frequency with which samples are taken from the image for making predictions. For instance, a sample
        frequency of 1000 corresponds to 1 sample per 1000 pixels of the input image.
    label_aggregator : function from list of floats to float
        The function that is used to combine the labels predicted by the predictor into a single label. By default,
        only the first label is used.
    preprocessors : list of functions from np.array to np.array
        A list of the image processing functions that are applied to the original image before starting the sampling and
        prediction phase. The preprocessors are applied in the order of the list.
    interpolator : subclass of visualization.pixel_interpolation.Interpolator
        The interpolator that is used to infer the pixels of the heat map based on the predictions made in the sampling
        and prediction phase.
    heat_map_resolution : int
        The resolution of the heat map. For instance, a resolution of 5 implies that squares of 5 by 5 pixels in the
        original image are condensed into 1 pixel in the heat map.
    """

    # set up the heat map
    height = image.shape[0]
    width = image.shape[1]
    heat_map = np.zeros((height // heat_map_resolution, width // heat_map_resolution))

    # pre-process image
    for preprocessor in preprocessors:
        image = preprocessor(image)
    print('pre-processing complete')

    # pad image
    sample_size = predictor.get_image_size()
    border = sample_size // 2
    padded_img = np.pad(image, ((border, border + 1), (border, border + 1), (0, 0)), 'edge')

    # draw random sample coordinates without replacement
    coordinates = np.random.choice(height * width, (height * width) // sample_frequency, replace=False)

    # make predictions
    progress = 0
    step_size = len(coordinates) // 10
    predictions = {}
    for i in range(0, len(coordinates)):
        # print progress
        if step_size > 0 and i % step_size == 0:
            print("{}% of predictions complete".format(progress))
            progress += 10

        # make prediction and add to the sample dictionary
        y = coordinates[i] // width
        x = coordinates[i] % width
        predictions[(y, x)] = label_aggregator(predictor.predict([padded_img[y:y + sample_size, x:x + sample_size]])[0])

    # pass predictions to the interpolator
    interpolator.fit(predictions)

    # interpolate missing predictions
    progress = 0
    step_size = ((width // heat_map_resolution) * (height // heat_map_resolution)) // 10
    i = 0
    for x in range(width // heat_map_resolution):
        for y in range(height // heat_map_resolution):
            # print progress
            if step_size > 0 and i % step_size == 0:
                print("{}% of heat map complete".format(progress))
                progress += 10
            i += 1

            # make interpolation
            heat_map[y, x] = interpolator.interpolate(y * heat_map_resolution, x * heat_map_resolution)

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