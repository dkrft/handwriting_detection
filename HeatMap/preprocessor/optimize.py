import cv2
import numpy as np
import time
import os
from matplotlib import pyplot as plt
from utils import show, rdbscan

"""this file is used to optimize the preprocessing, such that
machine written text is removed from the original image"""

# genetic optimizer:
from treibhaus import Treibhaus, RandomSearch
# handwriting detection:
from detection import *

filenames = ['sample1.png', 'page0002.jpg']
# filenames = ['sample1.png']
observations = [cv2.imread('../../data/raw/'.replace('/', os.sep) + filename) for filename in filenames]
expectations = [cv2.imread('../../data/labels/'.replace('/', os.sep) + filename) for filename in filenames]

repititions = 1

# bad angle autodetection, don't use:
# observation, angle = rotate(cv2.imread('samples' + os.sep + filename))
# expectation, _ = rotate(cv2.imread('labels' + os.sep + filename), angle)

# how to test the quality of the model
def test(model):
    total_fitness = 0
    for observation, expectation in zip(observations, expectations):
        prediction = model.detect_handwriting(observation.copy(), verbose=False, repititions=repititions)
        mask = (expectation < 200) | (prediction < 200)
        difference = expectation[mask] - prediction[mask]
        error = (difference**2).mean()

        # and take special care about the handwritten text being still there
        # after the model was applied. Look for pixels in the prediction that are lighter
        # than in the expectation, which means handwritten text got removed! :(
        error2 = expectation[prediction > expectation].mean()
        # so if there is a dark pixel in the expectation,
        # and a white pixel in the prediction -> higher error
        # and if there is like a light pixel in the expectation,
        # and a white pixel in the prediction -> only slightly higher error
        
        # large error -> low fitness
        total_fitness -= error
        
    # print(total_fitness)
    return total_fitness

# function that creates the model
def model(*params):
    return Handwriting_Preprocessor(*params)

# optimizing a single picture, 4 models 8 gens:
# original implemtation: 40s
# using dbscan: 20s
# downscaling before blur-threshold-masking: 7s

start = time.time()


# start optimizing
optimizer = RandomSearch(model, test,
    128,
    [(120, 255, int), # bw_threshold
    (220, 255, int), # density_threshold_2
    (0.15, 0.45, float), # density_threshold_1
    (15, 70, int), # filter_size_multiplicator_1
    (10, 90, int), # filter_size_multiplicator_2
    (0.1, 0.6, float), # vertical_filter_size_1
    (0.1, 0.5, float), # vertical_filter_size_2
    (30, 120, int), # min_line_length
    (3, 10, int), # max_line_gap # before canny edge detection, this was 10 - 30
    (3, 8, int), # long_line_factor
    (40, 180, float), # epsilon_v_1
    (5, 20, float), # epsilon_h_1
    (6, 15, float), # epsilon_v_2
    (0.2, 1.3, float), # epsilon_h_2
    (5, 16, int), # min_samples_1
    (1, 4, int)], # min_samples_2
    workers=os.cpu_count())
    
"""stopping_kriterion_gens=None, verbose=True, new_individuals=1,
   exploration_damping=4, workers=os.cpu_count(), learning_rate=0.1,
   keep_parents=1)"""

# print how long it took
end = time.time()
print(round(end - start, 3), 's')

# log the parameters array for the best model
best = optimizer.get_best_parameters()
print(best)

# best = [141, 234, 0.1500014435606156, 70, 10, 0.1, 0.1, 30, 10, 3, 40.95942374396594, 20.0, 11.957106015346124, 0.4539178429717946, 5, 2]

# recreate the best model and show it again
a = Handwriting_Preprocessor(*best)
for observation in observations:
    show(a.detect_handwriting(observation, verbose=True, repititions=repititions))

# save history, don't overwrite old files
# n = 1
# while os.path.isfile('histories/history_' + str(n) + '.npy'):
#     n += 1
# np.array(optimizer.history).dump('histories/history_' + str(n) + '.npy')
