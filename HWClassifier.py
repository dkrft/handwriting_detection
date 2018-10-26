import numpy as np
import tensorflow as tf
import random
import pickle
from sklearn.model_selection import train_test_split
from os import listdir

DEFAULT_IMAGE_SHAPE = [425, 300, 1]
DEFAULT_FILTER_SHAPE = [5, 5]
DEFAULT_FILTER_NUM = 10
DEFAULT_FULL_LAYER_SIZES = [100, 10]
DEFAULT_TRAINING_ITERATION_NUM = 500
DEFAULT_TRAINING_BATCH_SIZE = 20


def create_model(
        input_shape=DEFAULT_IMAGE_SHAPE,
        filter_shape=DEFAULT_FILTER_SHAPE,
        filter_num=DEFAULT_FILTER_NUM,
        full_layer_sizes=DEFAULT_FULL_LAYER_SIZES):
    # input templates
    x = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, 1], name='x')

    # cnn layers
    conv_layer = tf.layers.conv2d(x, filter_num, filter_shape,
            padding='same', activation=tf.nn.relu, name="conv_layer")
    pool_layer = tf.layers.max_pooling2d(conv_layer,
            pool_size=[2, 2], strides=[2, 2], padding='same', name='pool_layer')
    pool_layer_reshape = tf.reshape(pool_layer, [-1, pool_layer.shape[1:].num_elements()], name='pool_layer_reshape')
    full_layer_1 = tf.layers.dense(pool_layer_reshape, full_layer_sizes[0],
            activation=tf.nn.relu, name='full_layer_1')
    full_layer_2 = tf.layers.dense(full_layer_1, full_layer_sizes[1],
            activation=tf.nn.relu, name='full_layer_2')
    y_pred = tf.layers.dense(full_layer_2, 1, activation=None)

    # loss function for optimizer
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # accuracy function
    predictions = tf.cast(tf.nn.sigmoid(y_pred) > 0.5, np.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), np.float32))
    return {'x': x, 'y_true': y_true, 'y_pred': y_pred, 'optimizer': optimizer, 'loss': loss, 'accuracy': accuracy}


def create_training_data(image_shape=DEFAULT_IMAGE_SHAPE):
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        # get list of filenames and labels
        filenames = ['training_data/hw/' + file for file in listdir('training_data/hw')]
        labels = [1.0 for i in range(len(listdir('training_data/hw')))]
        filenames += ['training_data/no_hw/' + file for file in listdir('training_data/no_hw')]
        labels += [0.0 for i in range(len(listdir('training_data/no_hw')))]

        # split into training and test stets
        filenames_train, filenames_test, labels_train, labels_test = train_test_split(filenames, labels,
            test_size=0.30, random_state=42)

        # function to decode jpeg files
        def jpeg_decoder(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=image_shape[2])
            image_resized = tf.image.resize_images(image_decoded, [image_shape[0], image_shape[1]])
            image = tf.cast(image_resized, tf.float32)
            return image

        # create data sets
        x_train = [session.run(jpeg_decoder(filename)) for filename in filenames_train]
        x_test = [session.run(jpeg_decoder(filename)) for filename in filenames_test]
        y_train = [[x] for x in labels_train]
        y_test = [[x] for x in labels_test]
    session.close()
    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}


def train_cnn(session, model, training_data,
        training_iteration_num= DEFAULT_TRAINING_ITERATION_NUM,
        training_batch_size= DEFAULT_TRAINING_BATCH_SIZE):
    # reset cnn
    session.run(tf.global_variables_initializer())
    # train cnn
    for i in range(training_iteration_num):
        # select a random batch from training data
        batch_indices = random.sample(range(len(training_data['x_train'])), training_batch_size)
        x_train_batch = [training_data['x_train'][i] for i in batch_indices]
        y_train_batch = [training_data['y_train'][i] for i in batch_indices]
        # perform an optimization step
        session.run(model['optimizer'],
                feed_dict={model['x']: x_train_batch, model['y_true']: y_train_batch})
        # evaluate and print performance
        if (i % 10 == 0) or (i == training_iteration_num - 1):
            # training data
            loss = session.run(model['loss'],
                    feed_dict={model['x']: training_data['x_train'], model['y_true']: training_data['y_train']})
            accuracy = session.run(model['accuracy'],
                    feed_dict={model['x']: training_data['x_train'], model['y_true']: training_data['y_train']})
            print('Training Data: loss = %0.5f, accuracy = %0.5f' % (loss, accuracy))
            #test data
            loss = session.run(model['loss'],
                    feed_dict={model['x']: training_data['x_test'], model['y_true']: training_data['y_test']})
            accuracy = session.run(model['accuracy'],
                    feed_dict={model['x']: training_data['x_test'], model['y_true']: training_data['y_test']})
            print('Test Data: loss = %0.5f, accuracy = %0.5f' % (loss, accuracy))


def main():
    # create computational graph and session
    session = tf.Session()

    # create cnn model
    model = create_model()
    print('model created')

    # uncomment the following code block to create and store the data set from files in training data
    #training_data = create_training_data()
    #with open('training_data/data_set.pkl', 'wb+') as f:
    #    pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)
    #print('training data created and saved')

    # load training data from from pickle file
    with open('training_data/data_set.pkl', 'rb') as f:
        training_data = pickle.load(f)
    print('training data loaded')

    # train cnn
    train_cnn(session, model, training_data)


if __name__ == '__main__':
    main()