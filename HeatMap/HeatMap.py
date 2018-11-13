import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
#from preprocessor import Handwriting_Preprocessor
import HWInterface
import time


def nearest_neighbour_heat_map(img, sample_size=150, skip_param=1000, heat_map_scale=10):
    # create heat map
    height = img.shape[0]
    width = img.shape[1]
    heat_map = np.zeros((height // heat_map_scale, width // heat_map_scale))

    # pad image
    border = sample_size // 2
    padded_img = np.pad(img, ((border, border + 1),(border, border + 1), (0,0)), 'edge')

    # draw random sample coordinates without replacement
    coordinates = np.random.choice(height * width, (height * width) // skip_param, replace=False)

    # load cnn
    cnn = HWInterface.HWInterface(model_path='../model_archive/11_07', model_index=1000)
    print('CNN loaded')

    # make predictions
    progress = 0
    predictions = {}
    for i in range(0, len(coordinates)):
        # print status update
        if i % (len(coordinates) // 10) == 0:
            print("{}% of predictions complete".format(progress))
            progress += 10
        # make prediction and write to heat map
        y = coordinates[i] // width
        x = coordinates[i] % width
        predictions[(y,x)] = cnn.predict([padded_img[y:y + sample_size, x:x + sample_size]])[0][0]

    # create kdtree
    kdtree = scipy.spatial.KDTree([key for key in predictions.keys()])

    # make nearest neighbour predictions
    progress = 0
    i = 0
    for x in range(width // heat_map_scale):
        for y in range(height // heat_map_scale):
            if i % (((width // heat_map_scale) * (height // heat_map_scale)) // 10) == 0:
                print("{}% of heatmap complete".format(progress))
                progress += 10
            i += 1

            _, neighbour_indices = kdtree.query([(y * heat_map_scale, x * heat_map_scale)], k=12)
            neighbours = [kdtree.data[i] for i in neighbour_indices[0]]
            neighbour_values = [predictions[(neighbour[0], neighbour[1])] for neighbour in neighbours]
            neighbour_values.sort()
            heat_map[y,x] = neighbour_values[3]

    return heat_map


def nearest_neighbour_heat_map_mp(img, sample_size=150, heat_map_scale=5, prediction_size=10):
    # create heat map
    height = img.shape[0]
    width = img.shape[1]
    stride_y = height // (height // sample_size)
    stride_x = width // (width // sample_size)
    heat_map = np.zeros((height // heat_map_scale, width // heat_map_scale))

    # padd image
    border = sample_size // 2
    padded_img = np.pad(img, ((border, border + 1),(border, border + 1), (0,0)), 'edge')

    # load cnn
    cnn = HWInterface.HWInterface(model_path='../model_archive/11_13', model_index=300)
    print('CNN loaded')

    # make predictions
    progress = 0
    i = 0
    predictions = {}
    for y in range(0, (height // stride_y) * stride_y, stride_y):
        for x in range(0, (width // stride_x) * stride_x, stride_x):
            # print status update
            if i % (((height // stride_y) * (width // stride_x)) // 10) == 0:
                print("{}% of predictions complete".format(progress))
                progress += 10
            i += 1
            # make prediction and write to heat map
            prediction = cnn.predict([padded_img[y:y + sample_size, x:x + sample_size]])[0]
            for j in range(len(prediction)):
                predictions[(y + (sample_size // prediction_size) * (j // prediction_size),
                             x + (sample_size // prediction_size) * (j % prediction_size))] = prediction[j]

    ## create kdtree
    kdtree = scipy.spatial.KDTree([key for key in predictions.keys()])

    ## make nearest neighbour predictions
    progress = 0
    i = 0
    for x in range(width // heat_map_scale):
        for y in range(height // heat_map_scale):
            if i % (((width // heat_map_scale) * (height // heat_map_scale)) // 10) == 0:
                print("{}% of heatmap complete".format(progress))
                progress += 10
            i += 1

            _, neighbour_indices = kdtree.query([(y * heat_map_scale, x * heat_map_scale)], k=12)
            neighbours = [kdtree.data[i] for i in neighbour_indices[0]]
            neighbour_values = [predictions[(neighbour[0], neighbour[1])] for neighbour in neighbours]
            neighbour_values.sort()
            heat_map[y,x] = neighbour_values[3]

    return heat_map


def stride_heat_map(img, preprocess=False):
    """
    Takes raw image, returns heat map.
    parameter
    ---------
    input : uint8 numpy.ndarray
        2D numpy array of floats (BW) or of 3-tuples of floats (RGB).
        Scanned document that might or might not contain handwriting
    """

    # img = Handwriting_Preprocessor().detect_handwriting(img)

    # show(img)

    # divide img into chunks
    stride = 50

    # the network has 150x150
    # input neurons
    w = 150
    h = 150

    # add padding to img if wanted
    padding = 0  # TODO will modify img

    # hm = heat map
    hm_w = int((img.shape[1] - w) / stride) + 1
    hm_h = int((img.shape[0] - h) / stride) + 1

    assert hm_w > 0
    assert hm_h > 0

    # Classify each chunk and write
    # results into heat map.
    # Note, that the img objects are of shape (height, width)
    # opencv resizes and such however use (width, height) for their parameters and stuff
    # (no need to transpose, but it means just don't use img.shape on opencv params)
    heat_map = np.zeros((hm_h, hm_w))

    predictor = HWInterface.HWInterface(1000)

    print('CNN loaded')

    y_pos = 0
    while y_pos < hm_h:

        x_pos = 0
        while x_pos < hm_w:

            a = y_pos * stride
            b = x_pos * stride

            chunk = img[a:h + a, b:w + b]

            # skip everything that has too few dark pixels
            # if (chunk < chunk.mean()).astype(int).sum() / (150*150) > 0.015:
            # if chunk.min() < (chunk.max() - 150):
            if (255 - chunk).sum() > 255 * 500:
                
                c = predictor.predict([chunk])
                heat_map[y_pos, x_pos] = c[0]

            x_pos += 1

        # show 10% steps:
        if int(hm_h / 10) > 0 and y_pos % int(hm_h / 10) == 0:
            print(y_pos + 1, '/', hm_h, 'done')

        y_pos += 1

    heat_map = surroundings_noise_remover(heat_map)

    return heat_map


def surroundings_noise_remover(heat_map, passes=2):
    """Removes noise"""
    
    for _ in range(passes):

        heat_map_noise_removed = heat_map.copy()

        for x in range(0, heat_map.shape[1]):
            for y in range(0, heat_map.shape[0]):

                # the 7 surrounding pixels including itself
                # need to exceed a sum of 4, otherwise it's
                # considered noise.
                val = []

                if x > 0:
                    val += [heat_map[y    , x - 1]]
                    if y > 0:
                        val += [heat_map[y - 1, x - 1]]
                    if y < heat_map.shape[0] - 1:
                        val += [heat_map[y + 1, x - 1]]

                if x < heat_map.shape[1] - 1:
                    val += [heat_map[y    , x + 1]]
                    if y > 0:
                        val += [heat_map[y - 1, x + 1]]
                    if y < heat_map.shape[0] - 1:
                        val += [heat_map[y + 1, x + 1]]

                if y > 0:
                    val += [heat_map[y - 1, x    ]]
                if y < heat_map.shape[0] - 1:
                    val += [heat_map[y + 1, x    ]]

                # that's the center:
                val += [heat_map[y    , x    ]]

                if sum(val)/len(val) < 5/len(val):
                    heat_map_noise_removed[y, x] = 0

        # for the next pass
        heat_map = heat_map_noise_removed

    return heat_map_noise_removed


def show(img, heat_map):
    height, width, _ = img.shape
    plt.imshow(img)
    plt.imshow(cv2.resize(heat_map, (width, height), interpolation=cv2.INTER_NEAREST),
               cmap=plt.cm.viridis,
               alpha=.6,
               interpolation='bilinear')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    img = cv2.imread('../training_data/test_3.jpg')

    start = time.time()

    heat_map = nearest_neighbour_heat_map_mp(img)
    # heat_map = stride_heat_map(img)
    
    end = time.time()
    print(round(end - start, 3), 'Seconds')

    show(img, heat_map)


if __name__ == '__main__':
    main()
