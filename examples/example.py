import hwdetect
import cv2
import time
from sklearn.neighbors import KNeighborsRegressor
from hwdetect.visualization.interpolation import NearestNeighbour
from hwdetect.utils import show, get_path

# all values have standard values and don't have to be provided
# (except the img on which you want to predict of course):

# if predictor not provided, will use pretrained
# model from our repository

img = cv2.imread(get_path('examples/example_data/medium2.jpg'))

# show(img)
# show(hwdetect.preprocessor.Bandpass().preprocess(img))
# quit()

start = time.time()

heatmap = hwdetect.visualization.create_heat_map(img,
            preprocessors=[hwdetect.preprocessor.Bandpass()],
            sampler=hwdetect.visualization.sampler.Stride(),
            predictor=hwdetect.neural_network.Predictor(gpu=1),
            interpolator=KNeighborsRegressor())

print(round(time.time() - start, 3), 'Seconds')

hwdetect.visualization.plot_heat_map(img, heatmap)
