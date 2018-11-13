import pickle
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import os

def print_best_iteration():
    with open('trained_model/loss_accuracy_log.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data)
    loss_min = (-1,999999)
    for i in range(len(data['loss_train'])):
        if data['loss_train'][i][1] < loss_min[1]:
            loss_min = data['loss_train'][i]
    print("min training loss of {} in iteration {}".format(loss_min[1], loss_min[0]))

    accuracy_max = (-1,0)
    for i in range(len(data['accuracy_train'])):
        if data['accuracy_train'][i][1] > accuracy_max[1]:
            accuracy_max = data['accuracy_train'][i]
    print("max training accuracy of {} in iteration {}".format(accuracy_max[1], accuracy_max[0]))

    loss_min = (-1,999999)
    for i in range(len(data['loss_test'])):
        if data['loss_test'][i][1] < loss_min[1]:
            loss_min = data['loss_test'][i]
    print("min test loss of {} in iteration {}".format(loss_min[1], loss_min[0]))

    accuracy_max = (-1,0)
    for i in range(len(data['accuracy_test'])):
        if data['accuracy_test'][i][1] > accuracy_max[1]:
            accuracy_max = data['accuracy_test'][i]
    print("max test accuracy of {} in iteration {}".format(accuracy_max[1], accuracy_max[0]))

def show_loss_log():
    with open('trained_model/loss_accuracy_log.pkl', 'rb') as f:
        data = pickle.load(f)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss on Training and Test Data')
    plt.plot([log[0] for log in data['loss_train']], [log[1] for log in data['loss_train']], label="Training")
    plt.plot([log[0] for log in data['loss_test']], [log[1] for log in data['loss_test']], label="Test")
    plt.legend()
    plt.show()

def show_accuracy_log():
    with open('trained_model/loss_accuracy_log.pkl', 'rb') as f:
        data = pickle.load(f)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on Training and Test Data')
    plt.plot([log[0] for log in data['accuracy_train']], [log[1] for log in data['accuracy_train']], label="Training")
    plt.plot([log[0] for log in data['accuracy_test']], [log[1] for log in data['accuracy_test']], label="Test")
    plt.legend()
    plt.show()


def precision(model_index, data_set):
    session = tf.Session()

    # load model
    cwd = os.getcwd()
    path = os.path.join(cwd, 'trained_model/hw_classifier-0.meta')
    saver = tf.train.import_meta_graph(path)
    saver.restore(session, 'trained_model/hw_classifier-' + str(model_index))
    print('model loaded')

    # load data
    with open('../training_data/' + data_set, 'rb') as f:
        data = pickle.load(f)
    print('data loaded')

    # get in and output tensors
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y_pred/BiasAdd:0')

    # count false prediction
    false_pos = 0
    false_neg = 0
    pos = 0
    neg = 0
    for j in range(0, len(data['x_test']), 100):
        x_batch = data['x_test'][j:j + 100]
        y_batch = data['y_test'][j:j + 100]
        feed_dict = {x: x_batch}
        prediction = [int((1 / (1 + math.exp(-y_val))) > 0.75) for y_val in session.run(y, feed_dict)]
        for i in range(len(prediction)):
            if y_batch[i][0] == 0:
                neg += 1
                if prediction[i] == 1:
                    false_pos += 1
            if y_batch[i][0] == 1:
                pos += 1
                if prediction[i] == 0:
                    false_neg += 1
    print('prob. of false prediction is: %0.5f' % ((1.0 * (false_pos + false_neg)) / (1.0 * (pos + neg))))
    print('prob. of false positive is: %0.5f' % ((1.0 * false_pos) / (1.0 * pos)))
    print('prob. of false negative is: %0.5f' % ((1.0 * false_neg) / (1.0 * neg)))
    session.close()


if __name__ == '__main__':
    print_best_iteration()
    #show_loss_log()
    #show_accuracy_log()
    #print(precision(1000, 'ariel_26-10_5959.pkl'))