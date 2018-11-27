"""Module for creating and training neural networks.

This module provides an interface for creating and training simple models of convolutional neural networks for image
classification. The initial model and the progress that is made during the training phase are automatically saved to the
file system. The module also provides tools for analyzing the models performance during the training process.

Example
-------

Create and train a neuronal network to decide whether an image is light or dark.

    >>> import numpy as np
    >>> import os
    >>> from data.training_data import TrainingData
    >>> from neural_network import model
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
    >>> # create a model of a convolutional neural network to classify the brightness of 100 by 100 pixel images
    >>> # with 3 color channels using 1 label
    >>> model.create(model_directory, 100, 3, 1)
    >>>
    >>> # train the newly created model for 15 iterations, save it after every 2 iterations and keep only the 5 most
    >>> # recent iterations in the file system at all times
    >>> model.train(model_directory, training_data, max_iterations=15, save_frequency=2, max_save=5)

Retrain the previous model with a different data set.

    >>> # create a new data set for training with noisy images
    >>> black_noisy_image = np.random.randint(0, high=64, size=(100, 100, 3))
    >>> white_noisy_image = np.random.randint(192, high=256, size=(100, 100, 3))
    >>> noisy_training_data = TrainingData([black_noisy_image, white_noisy_image], [[0], [1]], [grey_image], [[1]])
    >>>
    >>> # retrain the model obtained in iteration 8 for another 10 iterations on the new data set
    >>> model.train(model_directory, training_data,
    >>>             initial_iteration=8, max_iterations=10, save_frequency=2, max_save=5)

Plot the models performance during training.

    >>> # plot the loss function and accuracy over time
    >>> model.plot_training(model_directory, metric='loss', saved_only=False)
    >>> model.plot_training(model_directory, metric='accuracy', saved_only=False)
    >>>
    >>> # get the iterations that exhibited the lowest loss and highest accuracy on to the test data during training
    >>> # only consider iterations that are stored and can be loaded from the file system
    >>>  print('iteration with lowest loss: ' + str(model.get_best_recorded_iteration(model_directory,
    >>>                                                                               data_set='test',
    >>>                                                                               metric='loss',
    >>>                                                                               saved_only=True)))
    >>> print('iteration with highest accuracy: ' + str(model.get_best_recorded_iteration(model_directory,
    >>>                                                                                   data_set='test',
    >>>                                                                                   metric='accuracy',
    >>>                                                                                   saved_only=True)))

"""

__author__ = "Dennis Kraft"
__version__ = "1.0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random
import pickle
import os
import re

DEFAULT_ITERATION_NUM = 6000
DEFAULT_TRAINING_BATCH_SIZE = 100
DEFAULT_SAVE_INTERVAL = 100
DEFAULT_POS_WEIGHT = 1


def create(
        model_directory,
        image_size,
        image_channels,
        label_size,
        filter_size=5,
        filter_num=15,
        first_full_layer_size=250,
        second_full_layer_size=50,
        third_full_layer_size=10):
    """Create a new model of a convolutional neural network and save it to the file system.

    Parameters
    ----------
    model_directory : str
        The path to the directory in which the model is saved. The path must be defined in absolute terms. Furthermore
        it should lead to an empty directory. If the specified directory does not exist in the file system already, a
        new directory is created.
    image_size : int
        The width and height of the images accepted by the neural network. The sampler will read this value in order
        to sample from the input image.
    image_channels : int
        The number of channels of the images accepted by the neural network.
    label_size : int
        The number of labels used to classify the images passed to the neural network
    filter_size : int, optional
        The width and height of the filters used in the convolution layers.
    filter_num : int, optional
        The number of filters per convolution layer.
    first_full_layer_size : int, optional
        The number of nodes in the first fully connected layer.
    second_full_layer_size : int, optional
        The number of nodes in the second fully connected layer.
    third_full_layer_size : int, optional
        The number of nodes in the third fully conected layer.
    """

    # create the computational graph
    graph = tf.Graph()
    with graph.as_default():
        # with tf.device('/device:XLA_GPU:0'):

        # create input templates
        x = tf.placeholder(tf.float32, shape=[
                           None, image_size, image_size, image_channels], name='x')
        y_true = tf.placeholder(
            tf.float32, shape=[None, label_size], name='y_true')
        drop_out_rate = tf.placeholder(
            tf.float32, shape=[1], name='drop_out_rate')
        positive_weight = tf.placeholder(
            tf.float32, shape=[1], name='positive_weight')

        # create the layers of the neural network
        convolution_layer_1 = tf.layers.conv2d(x, filter_num, (filter_size, filter_size),
                                               padding='same', activation=tf.nn.relu, name="convolution_layer_1")
        pool_layer_1 = tf.layers.max_pooling2d(convolution_layer_1,
                                               pool_size=[2, 2], strides=[2, 2], padding='same', name='pool_layer_1')
        convolution_layer_2 = tf.layers.conv2d(pool_layer_1, filter_num, (filter_size, filter_size),
                                               padding='same', activation=tf.nn.relu, name="convolution_layer_2")
        pool_layer_2 = tf.layers.max_pooling2d(convolution_layer_2,
                                               pool_size=[2, 2], strides=[2, 2], padding='same', name='pool_layer_2')
        pool_layer_reshape = tf.reshape(pool_layer_2, [-1, pool_layer_2.shape[1:].num_elements()],
                                        name='pool_layer_reshape')
        drop_out_layer_1 = tf.layers.dropout(
            pool_layer_reshape, rate=drop_out_rate, name='drop_out_layer_1')
        full_layer_1 = tf.layers.dense(drop_out_layer_1, first_full_layer_size,
                                       activation=tf.nn.relu, name='full_layer_1')
        drop_out_layer_2 = tf.layers.dropout(
            full_layer_1, rate=drop_out_rate, name='drop_out_layer_2')
        full_layer_2 = tf.layers.dense(drop_out_layer_2, second_full_layer_size,
                                       activation=tf.nn.relu, name='full_layer_2')
        drop_out_layer_3 = tf.layers.dropout(
            full_layer_2, rate=drop_out_rate, name='drop_out_layer_3')
        full_layer_3 = tf.layers.dense(drop_out_layer_3, third_full_layer_size,
                                       activation=tf.nn.relu, name='full_layer_3')
        y_prediction = tf.layers.dense(
            full_layer_3, label_size, activation=None, name='y_prediction')

        # specify the loss function used by the optimizer during training
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=y_prediction,
                                                                       targets=y_true,
                                                                       pos_weight=positive_weight), name='loss')
        tf.train.AdamOptimizer().minimize(loss, name='optimizer')

        # specify the accuracy function used for evaluating performance during
        # training
        predictions = tf.cast(tf.nn.sigmoid(y_prediction) > 0.5, np.float32)
        tf.reduce_mean(tf.cast(tf.equal(predictions, y_true),
                               np.float32), name='accuracy')

        # initialize the model and save it to the file system
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            tf.train.Saver().save(session, model_directory + '/model', global_step=0)

    # create a log file and save it to the file system
    with open(model_directory + '/log.pkl', 'wb+') as f:
        pickle.dump({'iterations': [],
                     'losses_train': [],
                     'accuracies_train': [],
                     'losses_test': [],
                     'accuracies_test': [],
                     'image_size': image_size,
                     'image_channels': image_channels,
                     'label_size': label_size}, f, pickle.HIGHEST_PROTOCOL)

    print('model created')


def train(model_directory, training_data,
          initial_iteration='last',
          max_iterations=1000,
          batch_size=150,
          drop_out_rate=0.0,
          positive_weight=1.0,
          save_frequency=100,
          max_save=50):
    """Train the model of a convolutional neural network with the provided data set.

    Parameters
    ----------
    model_directory : str
        The path to the directory from which the model is loaded and to which the trained models are saved. The path
        must be defined in absolute terms.
    training_data : {data.training_data.TrainingData, str}
        The data set used for training. The data can either be passed as a training data object or loaded from a pickle
        file that contains a training data object. In the latter case, the absolute path to the file needs to be passed
        as a string.
    initial_iteration : {int, str}, optional
        The iteration from which the training process is started. Note that all records of iterations that where
        performed subsequent to this iteration are deleted from the file system and log file. If the string ``last`` is
        passed as an argument, the the most recent iteration saved in the file system is chosen.
    max_iterations: int, option
        The total number of iterations performed during training. If the training is interrupted at any point in time,
        it is possible to access all models that were saved before the current iteration via the file system. This
        feature can be used to stop the training process early if desired.
    batch_size: int, option
        The number of samples used for optimizing the model in a single iteration of the training process. The samples
        of each batch are selected randomly from the training data. Reducing the batch size may avoid memory issues
        during training.
    drop_out_rate : float, optional
        The probability with which each node of the fully connected layer is temporarily removed from the network
        during a training iteration. Increasing the dropout rate may reduce over fitting.
    positive_weight: float, optional
        The weight assigned to positive labels in the loss function. Choosing a value greater than 1 increases the
        recall with respect to positive labels while a value less than 1 increases the precision. This parameter can
        also be used to handel unbalanced training data.
    save_frequency: int, optional
        The frequency with which the iterations are saved to the file system. For instance, a save frequency of 100
        means that the model is saved after every 100 iterations. The last iteration of the training process is saved
        to the file system independent of the save frequency. Each time an iteration is saved to the file system, a
        brief update on the current loss and accuracy of the model with respect to the training data set is printed.
    max_save: int, optional
        The maximum number of iterations that are saved during this training process. To satisfy the max keep
        constraint, iterations created during the current training phase are deleted in decreasing order of age
        whenever necessary.
    """

    # create computational graph and session
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as session:

            # set initial iteration to appropriate value if ``last`` was passed
            # as the argument
            if initial_iteration == 'last':
                initial_iteration = max(get_recorded_iterations(
                    model_directory, saved_only=True))

            # load training data from file if necessary
            if type(training_data) == str:
                with open(training_data, 'rb') as f:
                    training_data = pickle.load(f)

            # load model from file
            loader = tf.train.import_meta_graph(
                model_directory + '/model-0.meta')
            loader.restore(session, model_directory +
                           '/model-' + str(initial_iteration))

            # delete iterations greater than the initial iteration before
            # starting the training
            iterations_to_delete = get_recorded_iterations(
                model_directory, saved_only=False)
            iterations_to_delete = set([iteration for iteration in iterations_to_delete
                                        if iteration > initial_iteration])
            delete(model_directory, iterations_to_delete)

            # get tensors used for passing training data and parameters to the
            # neural network
            x_tensor = graph.get_tensor_by_name('x:0')
            y_true_tensor = graph.get_tensor_by_name('y_true:0')
            drop_out_rate_tensor = graph.get_tensor_by_name('drop_out_rate:0')
            positive_weight_tensor = graph.get_tensor_by_name(
                'positive_weight:0')

            # get tensors used for reading predictions, accuracy and loss from
            # the neural network
            y_prediction_tensor = graph.get_tensor_by_name(
                'y_prediction/BiasAdd:0')
            loss_tensor = graph.get_tensor_by_name('loss:0')
            accuracy_tensor = graph.get_tensor_by_name('accuracy:0')

            # get optimizer
            optimizer = graph.get_operation_by_name('optimizer')

            # prepare saver
            saver = tf.train.Saver(max_to_keep=max_save)

            # training started
            print('training started')

            # start training
            for i in range(1, max_iterations + 1):

                # select a random batch from the training data
                batch_indices = random.sample(range(len(training_data.x_train)),
                                              min(len(training_data.x_train), batch_size))
                x_train_batch = [training_data.x_train[j]
                                 for j in batch_indices]
                y_train_batch = [training_data.y_train[j]
                                 for j in batch_indices]

                # perform an optimization step
                feed_dict = {x_tensor: x_train_batch,
                             y_true_tensor: y_train_batch,
                             drop_out_rate_tensor: [drop_out_rate],
                             positive_weight_tensor: [positive_weight]}
                session.run(optimizer, feed_dict=feed_dict)

                # evaluate the performance of the neural network and save its
                # current state to the file system
                if (i % save_frequency == 0) or (i == max_iterations):

                    # iterate through the training data in chunks to save memory
                    loss_train = 0.0
                    accuracy_train = 0.0
                    for j in range(0, len(training_data.x_train), batch_size):
                        x_eval_batch = training_data.x_train[j:j + batch_size]
                        y_eval_batch = training_data.y_train[j:j + batch_size]
                        feed_dict = {x_tensor: x_eval_batch,
                                     y_true_tensor: y_eval_batch,
                                     drop_out_rate_tensor: [drop_out_rate],
                                     positive_weight_tensor: [positive_weight]}
                        scaling_coefficient = (
                            1.0 * len(x_eval_batch)) / (1.0 * len(training_data.x_train))
                        loss_train += scaling_coefficient * \
                            session.run(loss_tensor, feed_dict=feed_dict)
                        accuracy_train += scaling_coefficient * \
                            session.run(accuracy_tensor, feed_dict=feed_dict)

                    # iterate through the test data set in chunks to save memory
                    loss_test = 0.0
                    accuracy_test = 0.0
                    for j in range(0, len(training_data.x_test), batch_size):
                        x_eval_batch = training_data.x_test[j:j + batch_size]
                        y_eval_batch = training_data.y_test[j:j + batch_size]
                        feed_dict = {x_tensor: x_eval_batch,
                                     y_true_tensor: y_eval_batch,
                                     drop_out_rate_tensor: [drop_out_rate],
                                     positive_weight_tensor: [positive_weight]}
                        scaling_coefficient = (
                            1.0 * len(x_eval_batch)) / (1.0 * len(training_data.x_test))
                        loss_test += scaling_coefficient * \
                            session.run(loss_tensor, feed_dict=feed_dict)
                        accuracy_test += scaling_coefficient * \
                            session.run(accuracy_tensor, feed_dict=feed_dict)

                    # add the results of the evaluation to the log file
                    with open(model_directory + '/log.pkl', 'rb') as f:
                        log = pickle.load(f)
                    log['iterations'].append(initial_iteration + i)
                    log['losses_train'].append(loss_train)
                    log['accuracies_train'].append(accuracy_train)
                    log['losses_test'].append(loss_test)
                    log['accuracies_test'].append(accuracy_test)
                    with open(model_directory + '/log.pkl', 'wb+') as f:
                        pickle.dump(log, f, pickle.HIGHEST_PROTOCOL)

                    # save the current state of the neural network to the file
                    # system
                    saver.save(session, model_directory + '/model',
                               global_step=initial_iteration + i, write_meta_graph=False)

                    # print information about the current state of the training
                    # process
                    print('training status update')
                    print('\t current iteration: %d' % (initial_iteration + i,))
                    print('\t training data loss: %0.5f' % (loss_train,))
                    print('\t training data accuracy: %0.5f' %
                          (accuracy_train,))
                    print('\t test data loss: %0.5f' % (loss_test,))
                    print('\t test data accuracy: %0.5f' % (accuracy_test,))

    print('training complete')


def delete(model_directory, iterations):
    """Delete the specified iterations from the file system and log file.

    Parameters
    ----------
    model_directory : str
        The path to the directory of the model from which the iterations shall be deleted. The path must be defined in
        absolute terms.
    iterations : set of int
        The iterations to delete from the file system and log files.
    """

    # delete from file system
    for filename in os.listdir(model_directory):
        match = re.search(r'model-(.+?)\.', filename)
        if match:
            iteration = int(match.group(1))
            if iteration in iterations:
                os.remove(model_directory + '/' + filename)

    # delete from log file
    with open(model_directory + '/log.pkl', 'rb') as f:
        log = pickle.load(f)
    log['losses_train'] = [log['losses_train'][i] for i in range(0, len(log['iterations']))
                           if log['iterations'][i] not in iterations]
    log['accuracies_train'] = [log['accuracies_train'][i] for i in range(0, len(log['iterations']))
                               if log['iterations'][i] not in iterations]
    log['losses_test'] = [log['losses_test'][i] for i in range(0, len(log['iterations']))
                          if log['iterations'][i] not in iterations]
    log['accuracies_test'] = [log['accuracies_test'][i] for i in range(0, len(log['iterations']))
                              if log['iterations'][i] not in iterations]
    log['iterations'] = [log['iterations'][i] for i in range(0, len(log['iterations']))
                         if log['iterations'][i] not in iterations]
    with open(model_directory + '/log.pkl', 'wb+') as f:
        pickle.dump(log, f, pickle.HIGHEST_PROTOCOL)


def get_recorded_iterations(model_directory, saved_only=False):
    """Create a list of the recorded iterations.

    Parameters
    ----------
    model_directory : str
        The path to the directory of the model from which the saved iterations are collected. The path must be defined
        in absolute terms.
    saved_only : bool, optional
        A flag indicating whether all recorded iterations are returned or only those that can be recovered from the file
        system. Note that not all recorded iterations are kept during training to reduce memory use.

    Returns
    -------
    list of int
        A list of integers corresponding to the iterations that are currently recorded in the specified directory. The
        returned list is ordered in chronologically starting with the oldest iteration.
    """

    # collect iterations from file system if saved only flag is set
    if saved_only:
        iterations = set([])
        for filename in os.listdir(model_directory):
            match = re.search(r'model-(.+?)\.', filename)
            if match:
                iterations.add(int(match.group(1)))
        iterations = list(iterations)
        iterations.sort()
        return iterations

    # retrieve iterations from log file and file system if saved only flag is
    # not set
    else:
        with open(model_directory + '/log.pkl', 'rb') as f:
            # add iteration 0 to the output since this is not recorded in the
            # log file)
            return [0] + pickle.load(f)['iterations']


def get_best_recorded_iteration(model_directory, data_set='test', metric='loss', saved_only=True):
    """Find the iteration that performed best during training.

    Parameters
    ----------
    model_directory : str
        The path to the directory of the model whose training records are considered. The path must be defined in
        absolute terms.
    data_set : str, optional
        The data set that is used to evaluate the performance of an iteration. Set to ``test`` to consider the test data
         or to ``train`` to consider the training data.
    metric : str, optional
        The metric that is used to evaluate the performance of an iteration. Set to ``loss`` to consider the loss
        function or to ``accuracy`` to consider the accuracy.
    saved_only : bool, optional
        A flag indicating whether all recorded iteration are considered or only those that can be recovered from the
        file system. Note that not all recorded iterations are kept during training to reduce memory use.

    Returns
    -------
    int
        The iteration with the best performance according to the specified parameters. In case of a tie, only the first
        iteration in chronological oder is returned.
    """

    # get the set of iterations that need to be considered and load the log file
    iterations = set(get_recorded_iterations(
        model_directory, saved_only=saved_only))
    with open(model_directory + '/log.pkl', 'rb') as f:
        log = pickle.load(f)

    # get iteration with the minimum loss
    if metric == 'loss':
        loss = float('inf')
        iteration = 0
        for i in range(0, len(log['iterations'])):
            if log['losses_' + data_set][i] < loss and log['iterations'][i] in iterations:
                loss = log['losses_' + data_set][i]
                iteration = log['iterations'][i]
        return iteration

    # get iteration with the maximum accuracy
    elif metric == 'accuracy':
        accuracy = 0
        iteration = 0
        for i in range(0, len(log['iterations'])):
            if log['accuracies_' + data_set][i] > accuracy and log['iterations'][i] in iterations:
                accuracy = log['accuracies_' + data_set][i]
                iteration = log['iterations'][i]
        return iteration


def plot_training(model_directory, metric='loss', saved_only=False):
    """Create and show a plot of the training performance of the specified model.

    Parameters
    ----------
    model_directory : str
        The path to the directory of the model whose training performance is plotted. The path must be defined in
        absolute terms.
    metric : str, optional
        The metric that is plotted. Set to ``loss`` to plot the loss function or to ``accuracy`` to plot the accuracy.
    saved_only : bool, optional
        A flag indicating whether all recorded iteration are plotted or only those that can be recovered from the file
        system. Note that not all recorded iterations are kept during training to reduce memory use.

    Returns
    -------
    int
        The iteration with the best performance according to the specified parameters. In case of a tie, only the first
        iteration in chronological oder is returned.
    """

    # get the iterations that need to be considered and load the log file
    iteration_list = get_recorded_iterations(
        model_directory, saved_only=saved_only)
    iteration_list = iteration_list[1:len(iteration_list)]
    iteration_set = set(iteration_list)
    with open(model_directory + '/log.pkl', 'rb') as f:
        log = pickle.load(f)

    plt.figure(figsize=(24, 12))

    # prepare plot
    plt.xlabel('Iteration')

    # plot loss
    if metric == 'loss':
        plt.ylabel('Loss')
        plt.title('Loss on Training and Test Data')
        plt.plot(iteration_list, [log['losses_train'][i] for i in range(0, len(log['iterations']))
                                  if log['iterations'][i] in iteration_set], label="Training", color='#60cec4', lw=4)
        plt.plot(iteration_list, [log['losses_test'][i] for i in range(0, len(log['iterations']))
                                  if log['iterations'][i] in iteration_set], label="Test", color="#a065b3", lw=4)
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
        plt.ylim(-0.1, 2.1)

    # plot accuracies
    elif metric == 'accuracy':
        plt.ylabel('Accuracy')
        plt.title('Accuracy on Training and Test Data')
        plt.plot(iteration_list, [log['accuracies_train'][i] for i in range(0, len(log['iterations']))
                                  if log['iterations'][i] in iteration_set], label="Training", color='#60cec4', lw=4)
        plt.plot(iteration_list, [log['accuracies_test'][i] for i in range(0, len(log['iterations']))
                                  if log['iterations'][i] in iteration_set], label="Test", color="#a065b3", lw=4)
        plt.ylim(0.5, 1.02)
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.025))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))

    plt.gca().xaxis.set_minor_locator(MultipleLocator(100))
    plt.gca().xaxis.set_major_locator(MultipleLocator(500))

    plt.legend()
    plt.show()


def get_image_size(model_directory):
    """Get the size of the images that are accepted by the specified model.

    Parameters
    ----------
    model_directory : str
        The path to the directory of the model that is considered. The path must be defined in absolute terms.

    Returns
    -------
    int
        The width and height of the images accepted by the model.
    """

    # load the log file and read out the size of the images accepted by the
    # model
    with open(model_directory + '/log.pkl', 'rb') as f:
        return pickle.load(f)['image_size']


def get_image_channels(model_directory):
    """Get the number of channels of the images that are accepted by the specified model.

    Parameters
    ----------
    model_directory : str
        The path to the directory of the model that is considered. The path must be defined in absolute terms.

    Returns
    -------
    int
        The number of channels of the images accepted by the model.
    """

    # load the log file and read out the number of channels of the images
    # accepted by the model
    with open(model_directory + '/log.pkl', 'rb') as f:
        return pickle.load(f)['image_channels']


def get_label_size(model_directory):
    """Get the number of labels used to classify the images passed to the model.

    Parameters
    ----------
    model_directory : str
        The path to the directory of the model that is considered. The path must be defined in absolute terms.

    Returns
    -------
    int
        The number of labels used to classify the images passed to the model.
    """

    # load the log file and read out the number of labels used to classify the
    # images passed to the model.
    with open(model_directory + '/log.pkl', 'rb') as f:
        return pickle.load(f)['label_size']
