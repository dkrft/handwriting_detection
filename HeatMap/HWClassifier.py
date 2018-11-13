import numpy as np
import tensorflow as tf
import random
import pickle
from sklearn.model_selection import train_test_split
import os

DEFAULT_FILE_NAME = 'hw_classifier'
DEFAULT_IMAGE_SIZE = 150
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_FILTER_SIZE = 5
DEFAULT_FILTER_NUM = 25
DEFAULT_FC_LAYER_1_SIZE = 450
DEFAULT_FC_LAYER_2_SIZE = 150
DEFAULT_FC_LAYER_3_SIZE = 50
DEFAULT_OUTPUT_SIZE = 100
DEFAULT_TRAINING_ITERATION_NUM = 10000
DEFAULT_TRAINING_BATCH_SIZE = 100
DEFAULT_SAVE_INTERVAL = 100
DEFAULT_POS_WEIGHT = 1


def create_model(
        image_size= DEFAULT_IMAGE_SIZE,
        image_channels= DEFAULT_IMAGE_CHANNELS,
        filter_size= DEFAULT_FILTER_SIZE,
        filter_num= DEFAULT_FILTER_NUM,
        fc_layer_1_size = DEFAULT_FC_LAYER_1_SIZE,
        fc_layer_2_size = DEFAULT_FC_LAYER_2_SIZE,
        fc_layer_3_size = DEFAULT_FC_LAYER_3_SIZE,
        output_size= DEFAULT_OUTPUT_SIZE,
        pos_weight= DEFAULT_POS_WEIGHT):

    # input templates
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, image_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, output_size], name='y_true')

    conv_layer_1 = tf.layers.conv2d(x, filter_num, (filter_size,filter_size),
                                  padding='same', activation=tf.nn.relu, name="conv_layer_1")
    pool_layer_1 = tf.layers.max_pooling2d(conv_layer_1,
                                         pool_size=[2, 2], strides=[2, 2], padding='same', name='pool_layer_1')
    drop_out_layer_1 = tf.layers.dropout(pool_layer_1, rate=0.1, name='drop_out_layer_1')
    conv_layer_2 = tf.layers.conv2d(drop_out_layer_1, filter_num, (filter_size, filter_size),
                                    padding='same', activation=tf.nn.relu, name="conv_layer_2")
    pool_layer_2 = tf.layers.max_pooling2d(conv_layer_2,
                                           pool_size=[2, 2], strides=[2, 2], padding='same', name='pool_layer_2')
    pool_layer_reshape = tf.reshape(pool_layer_2, [-1, pool_layer_2.shape[1:].num_elements()],
                                    name='pool_layer_reshape')
    drop_out_layer_2 = tf.layers.dropout(pool_layer_reshape, rate=0.5, name='drop_out_layer_2')

    full_layer_1 = tf.layers.dense(drop_out_layer_2, fc_layer_1_size,
                                   activation=tf.nn.relu, name='full_layer_1')
    drop_out_layer_3 = tf.layers.dropout(full_layer_1, rate=0.5, name='drop_out_layer_3')
    full_layer_2 = tf.layers.dense(drop_out_layer_3, fc_layer_2_size,
                                   activation=tf.nn.relu, name='full_layer_2')
    drop_out_layer_4 = tf.layers.dropout(full_layer_2, rate=0.5, name='drop_out_layer_4')
    full_layer_3 = tf.layers.dense(drop_out_layer_4, fc_layer_3_size,
                                   activation=tf.nn.relu, name='full_layer_3')

    y_pred = tf.layers.dense(full_layer_3, output_size, activation=None, name='y_pred')

    # loss function for optimizer
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                                   targets=y_true,
                                                                   pos_weight= pos_weight))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # accuracy function
    predictions = tf.cast(tf.nn.sigmoid(y_pred) > 0.5, np.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), np.float32))

    return {'x': x, 'y_true': y_true, 'y_pred': y_pred, 'optimizer': optimizer, 'loss': loss, 'accuracy': accuracy}


def compute_pos_weight(labels):
    pos_count = 0
    total_count = 0
    for label in labels:
        pos_count += sum(label)
        total_count += len(label)
    return (1.0 * total_count) / (1.0 * pos_count)


def train_cnn(session, model, training_data,
        training_iteration_num= DEFAULT_TRAINING_ITERATION_NUM,
        training_batch_size= DEFAULT_TRAINING_BATCH_SIZE,
        save_interval= DEFAULT_SAVE_INTERVAL):
    # reset cnn
    session.run(tf.global_variables_initializer())

    # prepare saver
    saver = tf.train.Saver(max_to_keep=100)
    cwd = os.getcwd()
    path = os.path.join(cwd, "trained_model/" + DEFAULT_FILE_NAME)

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
        if (i % save_interval == 0) or (i == training_iteration_num - 1):
            # training data
            loss = 0.0
            accuracy = 0.0
            # split data into chunks to save memory
            for j in range(0, len(training_data['x_train']), training_batch_size):
                x_check_batch = training_data['x_train'][j:j + training_batch_size]
                y_check_batch = training_data['y_train'][j:j + training_batch_size]
                scaler = (1.0 * len(x_check_batch)) / (1.0 * len(training_data['x_train']))
                loss += session.run(model['loss'],
                                    feed_dict={model['x']: x_check_batch, model['y_true']: y_check_batch}) * scaler
                accuracy += session.run(model['accuracy'],
                                        feed_dict={model['x']: x_check_batch, model['y_true']: y_check_batch}) * scaler
            loss_vec_train.append((i, loss))
            accuracy_vec_train.append((i, accuracy))
            print('Training Data: loss = %0.5f, accuracy = %0.5f' % (loss, accuracy))
            # test data
            loss = 0.0
            accuracy = 0.0
            # split data into chunks to save memory
            for j in range(0, len(training_data['x_test']), training_batch_size):
                x_check_batch = training_data['x_test'][j:j + training_batch_size]
                y_check_batch = training_data['y_test'][j:j + training_batch_size]
                scaler = (1.0 * len(x_check_batch)) / (1.0 * len(training_data['x_test']))
                loss += session.run(model['loss'],
                                    feed_dict={model['x']: x_check_batch, model['y_true']: y_check_batch}) * scaler
                accuracy += session.run(model['accuracy'],
                                        feed_dict={model['x']: x_check_batch, model['y_true']: y_check_batch}) * scaler
            loss_vec_test.append((i, loss))
            accuracy_vec_test.append((i, accuracy))
            print('Test Data: loss = %0.5f, accuracy = %0.5f' % (loss, accuracy))
            #save model
            if i == 0:
                saver.save(session, path, global_step=i)
            else:
                saver.save(session, path, global_step=i, write_meta_graph=False)
            #save stats
            with open('trained_model/loss_accuracy_log.pkl', 'wb+') as f:
                d = {'loss_train': loss_vec_train,
                     'accuracy_train': accuracy_vec_train,
                     'loss_test': loss_vec_test,
                     'accuracy_test': accuracy_vec_test}
                pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

def main():
    # load training data from from pickle file
    with open('../training_data/ariel_pixel_maps_877.pkl', 'rb') as f:
        training_data = pickle.load(f)
    print('training data loaded')
    pos_weight = compute_pos_weight(training_data['y_train'] + training_data['y_test'])

    # create computational graph and session
    session = tf.Session()

    # create cnn model
    model = create_model(pos_weight=pos_weight)
    print('model created')

    # convert Ariel's data set
    #with open('../training_data/pixel_maps_877.pkl', 'rb') as f:
    #    files = pickle.load(f)
    #    labels = pickle.load(f)
    #    labels = [[label] for label in labels]
    #x_train, x_test, y_train, y_test = train_test_split(files, labels, test_size=0.30, random_state=42)
    #training_data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
    #with open('../training_data/ariel_pixel_maps_877.pkl', 'wb+') as f:
    #    pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)
    #print('training data saved')

    # train cnn
    train_cnn(session, model, training_data)
    print('cnn trained')


if __name__ == '__main__':
    main()