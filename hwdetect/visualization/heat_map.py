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
    >>> chess_board_hm = heat_map.create_heat_map(chess_board, Predictor(model_directory), sample_stepsize=50)
    >>>
    >>> # show the heat map
    >>> heat_map.show_heat_map(chess_board, chess_board_hm)
"""

__author__ = "Dennis Kraft, Ariel Bridgeman, Tobias B <github.com/sezanzeb>"
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
import logging


# __name__ is hwdetect.visualization.heat_map
logger = logging.getLogger(__name__)


def create_heat_map(image, predictor,
                    label_aggregator=lambda labels: labels[0],
                    sampler=RandomGrid(),
                    preprocessors=[],
                    interpolator=NearestNeighbour(),
                    postprocessors=[],
                    heat_map_scale=10,
                    return_preprocessed=False):
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
        prediction phase. The preprocessors are applied in the order of the list. The goal of the preprocessors is to
        remove machine writing, so that the predictor can jump over those chunks that are white after preprocessing.
        The original image is used for prediction though.
    interpolator : class implementing visualization.pixel_interpolation.Interpolator
        The interpolator that is used to infer the pixels of the heat map based on the predictions made in the sampling
        and prediction phase.
    heat_map_scale : int
        The resolution of the heat map. For instance, a resolution of 5 implies that squares of 5 by 5 pixels in the
        original image are condensed into 1 pixel in the heat map.
    return_preprocessed : bool
        If True, will return a 2-tuple (heat_map, preprocessed)
        Default: False

    Returns
    -------
    np.array
        A two dimensional array representing the heat map. Note that the height and width of the array matches the
        height and width of the original image scaled down by the heat map resolution parameter.
    """

    original = image.copy()

    # set up the heat map
    height = image.shape[0]
    width = image.shape[1]
    heat_map_scale = min(heat_map_scale, min(width, height))
    heat_map = np.zeros((height // heat_map_scale, width // heat_map_scale))

    # preprocess image
    if len(preprocessors) > 0:
        logger.info('preprocessing...')
        for preprocessor in preprocessors:
            image = preprocessor.filter(image)

    # make predictions
    logger.info('predicting...')
    predictions = sampler.sample(image, predictor, label_aggregator, original)

    X_pred = [k for k in predictions]
    Y_pred = [predictions[k] for k in predictions]
    interpolator.fit(X_pred, Y_pred)

    # create list of tuples (y, x) for all the coordinates in the heatmap
    # [0] is height, [1] is width
    coords = np.concatenate(
        np.dstack(np.mgrid[:heat_map.shape[0], :heat_map.shape[1]]))
    coords_scaled = coords * heat_map_scale

    logger.info('interpolating...')
    values = interpolator.predict(coords_scaled)

    heat_map = values.reshape(heat_map.shape)

    if len(postprocessors) > 0:
        # postprocess heat_map
        logger.info('postprocessing...')
        for postprocessor in postprocessors:
            heat_map = postprocessor.filter(heat_map)

    logger.info('done')

    if return_preprocessed:
        return heat_map, image

    # default behaviour:
    return heat_map


def heat_map_to_img(heat_map):
    """Convert percentages from heat_map to a grayscale image

    Parameters
    ----------
    heat_map : np.array
        The heat map of an image.

    Returns
    ----------
    gray : np.array
        The grayscale image of the heat map.

    """
    if heat_map.max() <= 1:
        heat_map *= 255
    heat_map = heat_map.astype(np.uint8)
    return heat_map


def bounded_image(image, heat_map, bound_type="box", perc_thresh=0.90):
    """Create image with bounding boxes or contours using the heat map

    Parameters
    ----------
    image : np.array
        The image that is plotted.
    heat_map : np.array
        The heat map put on top of the image
    bound_type : str
        The string used to specify whether to use a "box" or "contour" for bounding.
    per_thresh ; float between 0 and 1
        The float to set the threshold for which grayscale values to set to black or white.

    Returns
    ----------
    np.array
        image with bounding objects

    """

    # make sure they are of the same height and width
    if image.shape[:2] != heat_map.shape[:2]:
        h, w = image.shape[:2]
        heat_map = cv2.resize(heat_map, (w, h))

    # convert heat map to image
    hm_img = heat_map_to_img(heat_map)

    # set threshold at % of 255
    limit = int(perc_thresh * 255)
    ret, thresh = cv2.threshold(hm_img, limit, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    if bound_type == "contour":
        bound_img = cv2.drawContours(image, contours, -1, (0, 0, 255), 10)

    elif bound_type == "box":
        bound_img = image
        for c in contours:
            # fit with rotated rectangle
            # ( center (x,y), (width, height), angle of rotation
            rect = cv2.minAreaRect(c)

            # if angle of rotated rectangle within 5 deg, draw normalrectangle
            if abs(rect[2]) < 5:
                x, y, w, h = cv2.boundingRect(c)
                # reject small samples; local fluctuations
                if w * h > 900:
                    bound_img = cv2.rectangle(
                        bound_img, (x, y), (x + w, y + h), (160, 101, 179), 10)
            else:
                w, h = rect[1]
                # reject small samples; local fluctuations
                if w * h > 900:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    bound_img = cv2.drawContours(
                        bound_img, [box], 0, (160, 101, 179), 10)
    return bound_img


def plot_heat_map(image, heat_map, bounding_box=None, bound_type="box", save_as=""):
    """Overlay an image with the specified heat map or bounding box and plot the result.

    Parameters
    ----------
    image : np.array
        The image that is plotted.
    heat_map : np.array
        The heat map put on top of the image
    bounding_box: bool
        The boolean to specify whether to use a bounding box or not
    bound_type: str
        The string used to specify whether to use a "box" or "contour" for bounding.
    """

    height, width, _ = image.shape
    hm = cv2.resize(heat_map, (width, height), interpolation=cv2.INTER_NEAREST)

    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if bounding_box:
        bound_img = bounded_image(RGB_img, hm, bound_type=bound_type)
        plt.imshow(bound_img, origin="upper", aspect='equal')

    else:
        plt.imshow(RGB_img)
        plt.imshow(hm,
                   cmap=plt.cm.viridis,
                   alpha=.6,
                   interpolation='bilinear',
                   vmin=0,
                   vmax=1,
                   origin="upper",
                   aspect='equal')
        cbar = plt.colorbar()
        # needed to fix striations that appear in color bar with assumed alpha
        # level
        cbar.set_alpha(1)
        cbar.draw_all()
    plt.xticks([])
    plt.yticks([])
    if save_as == "":
        plt.show()
    else:
        plt.savefig(save_as)
    plt.clf()
