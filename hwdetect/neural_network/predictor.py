"""Module for using a pre-trained model of a convolutional neural network to make predictions.

This module provides a predictor object. The predictor object loads a pre-trained model of a convolutional neural
network and use it to predict the labels of images.

Example
-------

Train a neuronal network and use ot to predict whether an image is light or dark.

    >>> import numpy as np
    >>> import os
    >>> from data.training_data import TrainingData
    >>> from neural_network import model
    >>> from neural_network.predictor import Predictor
    >>>
    >>> # create three 100 by 100 pixel images with 3 color channels and store them in a data set for training
    >>> black_image = np.full((100, 100, 3), 0)
    >>> white_image = np.full((100, 100, 3), 255)
    >>> grey_image = np.full((100, 100, 3), 192)
    >>> training_data = TrainingData([black_image, white_image], [[0], [1]], [grey_image], [[1]])
    >>>
    >>> # setup the path to the directory where the model of the neural network is saved
    >>> cwd = os.getcwd()
    >>> model_directory = os.path.join(cwd, "brightness_detector")
    >>>
    >>> # create and train the model
    >>> model.create(model_directory, 100, 3, 1)
    >>> model.train(model_directory, training_data, max_iterations=15, save_frequency=2, max_save=5)
    >>>
    >>> # create a noisy image for prediction
    >>> white_noisy_image = np.random.randint(192, high=256, size=(100, 100, 3))
    >>>
    >>> # create a predictor based on the most recent iteration of the trained model
    >>> predictor = Predictor(model_directory)
    >>>
    >>> # predict the probability with which the noisy white image is a bright image
    >>> print('predicted probability of a bright image: ' + str(predictor.predict([white_noisy_image])[0][0]))

Make another prediction using the untrained model of the neural network.

    >>> # create a predictor based on the initial model
    >>> untrained_predictor = Predictor(model_directory, model_iteration=0)
    >>>
    >>> # predict the probability with which the noisy white image is a bright image
    >>> print('predicted probability of a bright image: ' + str(untrained_predictor.predict([white_noisy_image])[0][0]))
"""

__author__ = "Dennis Kraft"
__version__ = "1.0"

import tensorflow as tf
import math
from hwdetect.neural_network import model


class Predictor:
    """An object to predict the labels of an image based on a pre-trained neural network model."""

    def __init__(self, model_directory, model_iteration='last'):
        """Create a predictor based on specified neural network.

        Parameters
        ----------
        model_directory : string
            The path to the directory of the neural network model that is going to be used for making predictions. The
            path must be defined in absolute terms.
        model_iteration : {int, string}, optional
            The training iteration that is going to be used for making predictions. If the string ``last`` is passed
            as an argument, the most recent iteration saved in the file system is used.
        """

        # keep model directory
        self.model_directory = model_directory

        # create the computational graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # set iteration to appropriate value if ``last`` was passed as the argument
            if model_iteration == 'last':
                model_iteration = max(model.get_recorded_iterations(model_directory, saved_only=True))

            # create session
            self.session = tf.Session()

            # load model from file
            loader = tf.train.import_meta_graph(model_directory + '/model-0.meta')
            loader.restore(self.session, model_directory + '/model-' + str(model_iteration))

            # get tensors used for passing images to the neural network and for reading out the prediction
            self.x_tensor = self.graph.get_tensor_by_name('x:0')
            self.y_tensor = self.graph.get_tensor_by_name('y_prediction/BiasAdd:0')

    def predict(self, images):
        """Make predictions for the specified images.

        Parameters
        ----------
        images : list of np.arrays
            The images for which predictions are made. Note that the shape of the image must match the shape of the
            input layer of neural network loaded into the predictor.

        Returns
        -------
        list of list of int
            A list containing the predicted probabilities for the labels of each image.
        """

        # create a dictionary to feed the images into the neural network
        feed_dict = {self.x_tensor: images}

        # make predictions and apply the sigmoid function to obtain probabilities
        # the input of the sigmoid function is capped by -10 from below and 10 from above to avoid numerical problems in
        # its calculation
        return [[1 / (1 + math.exp(-max(min(prediction[i], 10), -10))) for i in range(len(prediction))]
                for prediction in self.session.run(self.y_tensor, feed_dict)]

    def __del__(self):
        """Delete the predictor """

        # close session to free memory
        self.session.close()

    def get_image_size(self):
        """Get the size of the images that are accepted by the predictor.

        Returns
        -------
        int
            The width and height of the images accepted by the predictor.
        """

        return model.get_image_size(self.model_directory)

    def get_image_channels(self):
        """Get the number of channels of the images that are accepted by the predictor.

        Returns
        -------
        int
            The number of channels of the images accepted by the predictor.
        """

        return model.get_image_channels(self.model_directory)

    def get_label_size(self):
        """Get the number of labels per image returned by the predictor

        Returns
        -------
        int
            The number of labels per image returned by the predictor.
        """

        return model.get_labels_size(self.model_directory)