import tensorflow as tf
import os
import pickle
import math


class HWInterface:

    def __init__(self, model_path, model_index):
        # create session
        self.session = tf.Session()

        # load model
        cwd = os.getcwd()
        path = os.path.join(cwd, model_path + '/hw_classifier-0.meta')
        saver = tf.train.import_meta_graph(path)
        saver.restore(self.session, model_path + '/hw_classifier-' + str(model_index))


        # get in and output tensors
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name('x:0')
        self.y = self.graph.get_tensor_by_name('y_pred/BiasAdd:0')

    def predict(self, x):
        feed_dict = {self.x: x}
        return [[1 / (1 + math.exp(-y[i])) for i in range(len(y))] for y in self.session.run(self.y, feed_dict)]


#if __name__ == '__main__':
#    with open('training_data/ariel_26-10_2341.pkl', 'rb') as f:
#        training_data = pickle.load(f)
#
#    # Interface Usage example
#    interface = HWInterface(1000)
#    test_x = training_data['x_test'][:30]
#    test_y = training_data['y_test'][:30]
#    print(test_y)
#    print(interface.predict(test_x))
