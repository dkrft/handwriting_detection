import hwdetect
import cv2
import time
from sklearn.neighbors import KNeighborsRegressor
from hwdetect.visualization.interpolation import NearestNeighbour
from hwdetect.utils import show

# all values have standard values and don't have to be provided
# (except the img on which you want to predict of course):

# if predictor not provided, will use pretrained
# model from our repository

img = cv2.imread('example_data/medium1.jpg')
if img is None:
    img = cv2.imread('examples/example_data/medium1.jpg')

start = time.time()

heatmap = hwdetect.visualization.create_heat_map_2(img,
            preprocessors=[hwdetect.preprocessor.Threshold(), hwdetect.preprocessor.Bandpass()],
            sampler=hwdetect.visualization.sampler.Stride(stride=10),
            predictor=hwdetect.neural_network.Predictor(gpu=1),
            interpolator=KNeighborsRegressor())

print(round(time.time() - start, 3), 'Seconds')

hwdetect.utils.show(heatmap)

