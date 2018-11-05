import numpy as np
import tensorflow as tf
import random
import pickle
from sklearn.model_selection import train_test_split
import os

DEFAULT_IMAGE_SIZE = 150
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_FILTER_SIZE = 5
DEFAULT_FILTER_NUM = 15
DEFAULT_FC_LAYER_1_SIZE = 128
DEFAULT_FC_LAYER_2_SIZE = 32
DEFAULT_TRAINING_ITERATION_NUM = 1000
DEFAULT_TRAINING_BATCH_SIZE = 30


def create_model(
        image_size= DEFAULT_IMAGE_SIZE,
        image_channels= DEFAULT_IMAGE_CHANNELS,
        filter_size= DEFAULT_FILTER_SIZE,
        filter_num= DEFAULT_FILTER_NUM,
        fc_layer_1_size = DEFAULT_FC_LAYER_1_SIZE,
        fc_layer_2_size = DEFAULT_FC_LAYER_2_SIZE):

    # input templates
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, image_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, 1], name='x')

    conv_layer_1 = tf.layers.conv2d(x, filter_num, (filter_size,filter_size),
                                  padding='same', activation=tf.nn.relu, name="conv_layer_1")
    pool_layer_1 = tf.layers.max_pooling2d(conv_layer_1,
                                         pool_size=[2, 2], strides=[2, 2], padding='same', name='pool_layer_1')
    conv_layer_2 = tf.layers.conv2d(pool_layer_1, filter_num, (filter_size, filter_size),
                                    padding='same', activation=tf.nn.relu, name="conv_layer_2")
    pool_layer_2 = tf.layers.max_pooling2d(conv_layer_2,
                                           pool_size=[2, 2], strides=[2, 2], padding='same', name='pool_layer_2')
    pool_layer_reshape = tf.reshape(pool_layer_2, [-1, pool_layer_2.shape[1:].num_elements()],
                                    name='pool_layer_reshape')
    full_layer_1 = tf.layers.dense(pool_layer_reshape, fc_layer_1_size,
                                   activation=tf.nn.relu, name='full_layer_1')
    full_layer_2 = tf.layers.dense(full_layer_1, fc_layer_2_size,
                                   activation=tf.nn.relu, name='full_layer_2')
    y_pred = tf.layers.dense(full_layer_2, 1, activation=None)

    # loss function for optimizer
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # accuracy function
    predictions = tf.cast(tf.nn.sigmoid(y_pred) > 0.5, np.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), np.float32))
    return {'x': x, 'y_true': y_true, 'y_pred': y_pred, 'optimizer': optimizer, 'loss': loss, 'accuracy': accuracy}


def create_training_data(image_shape=[DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_CHANNELS]):
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        # get list of filenames and labels
        filenames = ['training_data/hw/' + file for file in os.listdir('training_data/hw')]
        labels = [1.0 for i in range(len(os.listdir('training_data/hw')))]
        filenames += ['training_data/no_hw/' + file for file in os.listdir('training_data/no_hw')]
        labels += [0.0 for i in range(len(os.listdir('training_data/no_hw')))]

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
    loss_vec_train = []
    loss_vec_test = []
    accuracy_vec_train = []
    accuracy_vec_test = []
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
            loss = 0.0
            accuracy = 0.0
            # split data into chunks to save memory
            for i in range(0, len(training_data['x_train']), training_batch_size):
                x_check_batch = training_data['x_train'][i:i + training_batch_size]
                y_check_batch = training_data['y_train'][i:i + training_batch_size]
                scaler = (1.0 * len(x_check_batch)) / (1.0 * len(training_data['x_train']))
                loss += session.run(model['loss'],
                                    feed_dict={model['x']: x_check_batch, model['y_true']: y_check_batch}) * scaler
                accuracy += session.run(model['accuracy'],
                                        feed_dict={model['x']: x_check_batch, model['y_true']: y_check_batch}) * scaler
            loss_vec_train.append(loss)
            accuracy_vec_train.append(accuracy)
            print('Training Data: loss = %0.5f, accuracy = %0.5f' % (loss, accuracy))
            # test data
            loss = 0.0
            accuracy = 0.0
            # split data into chunks to save memory
            for i in range(0, len(training_data['x_test']), training_batch_size):
                x_check_batch = training_data['x_test'][i:i + training_batch_size]
                y_check_batch = training_data['y_test'][i:i + training_batch_size]
                scaler = (1.0 * len(x_check_batch)) / (1.0 * len(training_data['x_test']))
                loss += session.run(model['loss'],
                                    feed_dict={model['x']: x_check_batch, model['y_true']: y_check_batch}) * scaler
                accuracy += session.run(model['accuracy'],
                                        feed_dict={model['x']: x_check_batch, model['y_true']: y_check_batch}) * scaler
            loss_vec_test.append(loss)
            accuracy_vec_test.append(accuracy)
            print('Test Data: loss = %0.5f, accuracy = %0.5f' % (loss, accuracy))

    print('loss function training set:')
    print(loss_vec_train)
    print('accuracy training set:')
    print(accuracy_vec_train)
    print('loss function test set:')
    print(loss_vec_test)
    print('accuracy test set:')
    print(accuracy_vec_test)


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

    # convert Ariel's data set
    #with open('training_data/equalSamp_26-10_538.pkl', 'rb') as f:
    #    files = pickle.load(f)
    #    labels = pickle.load(f)
    #    labels = [[label] for label in labels]
    #x_train, x_test, y_train, y_test = train_test_split(files, labels, test_size=0.30, random_state=42)
    #training_data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
    #with open('training_data/data_set_ariel_equal.pkl', 'wb+') as f:
    #    pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)
    #print('training data saved')

    # load training data from from pickle file
    with open('training_data/data_set_ariel_equal.pkl', 'rb') as f:
        training_data = pickle.load(f)
    print('training data loaded')

    # train cnn
    train_cnn(session, model, training_data)

    # save trained model
    # cwd = os.getcwd()
    # path = os.path.join(cwd, 'test')
    # tf.saved_model.simple_save(session, path,
    #            inputs={'x': model['x'], 'y_true': model['y_true']},
    #            outputs={'y_pred': model['y_pred']})
    #print('model saved')


if __name__ == '__main__':
    main()