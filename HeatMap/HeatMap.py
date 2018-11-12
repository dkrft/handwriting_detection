import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
from preprocessor import Handwriting_Preprocessor
import HWInterface
import time


def random_heatmap(img, sample_size= 150, skip_param= 1500, heat_map_scale= 15):
    # create heatmap
    height = img.shape[0]
    width = img.shape[1]
    heatmap = np.zeros((height // heat_map_scale, width // heat_map_scale))

    # padd image
    border = sample_size // 2
    padded_img = np.pad(img, ((border, border + 1),(border, border + 1), (0,0)), 'edge')

    # draw random sample coordinates without replacement
    coordinates = np.random.choice(height * width, (height * width) // skip_param, replace=False)

    # load cnn
    cnn = HWInterface.HWInterface(1000)
    print('CNN loaded')

    # make predictions
    progress = 0
    predictions = {}
    for i in range(0, len(coordinates)):
        # print status update
        if i % (len(coordinates) // 10) == 0:
            print("{}% of predictions complete".format(progress))
            progress += 10
        # fill buffer
        y = coordinates[i] // width
        x = coordinates[i] % width
        predictions[(y,x)] = cnn.predict([padded_img[y:y + sample_size, x:x + sample_size]])[0]

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
            heatmap[y,x] = neighbour_values[3]

    return heatmap

    # create buffer to pass to neuronal network
    #buffer_size = 200
    #buffer = np.empty((buffer_size, sample_size, sample_size, 3))

    # make predictions
    #progress = 0
    #for i in range(0, len(coordinates), buffer_size):
    #    # print status update
    #    if (i / buffer_size) % (len(coordinates) // (buffer_size * 10)) == 0:
    #        print("{}% of predictions complete".format(progress))
    #        progress += 10

    #    # fill buffer
    #    for j in range(i,min(i + buffer_size, len(coordinates))):
    #        x = coordinates[j] // width
    #        y = coordinates[j] % width
    #        buffer[j-i] = padded_img[x:x + sample_size, y:y + sample_size]
    #    # pass to predictor
    #    predictions = cnn.predict(buffer[0: min(buffer_size, len(coordinates) - i)])

    #    # write prediction into heat map
    #    for j in range(i, min(i + buffer_size, len(coordinates))):
    #        x = coordinates[j] // width
    #        y = coordinates[j] % width
    #        heatmap[x,y] = predictions[j-i]
    #return heatmap


def stride_heatmap(img, preprocess=False):
    """
    Takes raw image, returns heatmap.
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

    # hm = heatmap
    hm_w = int((img.shape[1] - w) / stride) + 1
    hm_h = int((img.shape[0] - h) / stride) + 1

    assert hm_w > 0
    assert hm_h > 0

    # Classify each chunk and write
    # results into heatmap.
    # Note, that the img objects are of shape (height, width)
    # opencv resizes and such however use (width, height) for their parameters and stuff
    # (no need to transpose, but it means just don't use img.shape on opencv params)
    heatmap = np.zeros((hm_h, hm_w))

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
                heatmap[y_pos, x_pos] = c[0]

            x_pos += 1

        # show 10% steps:
        if int(hm_h / 10) > 0 and y_pos % int(hm_h / 10) == 0:
            print(y_pos + 1, '/', hm_h, 'done')

        y_pos += 1

    heatmap = surroundings_noise_remover(heatmap)

    return heatmap


def surroundings_noise_remover(heatmap, passes=2):
    """Removes noise"""
    
    for _ in range(passes):

        heatmap_noise_removed = heatmap.copy()

        for x in range(0, heatmap.shape[1]):
            for y in range(0, heatmap.shape[0]):

                # the 7 surrounding pixels including itself
                # need to exceed a sum of 4, otherwise it's
                # considered noise.
                val = []

                if x > 0:
                    val += [heatmap[y    , x - 1]]
                    if y > 0:
                        val += [heatmap[y - 1, x - 1]]
                    if y < heatmap.shape[0] - 1:
                        val += [heatmap[y + 1, x - 1]]

                if x < heatmap.shape[1] - 1:
                    val += [heatmap[y    , x + 1]]
                    if y > 0:
                        val += [heatmap[y - 1, x + 1]]
                    if y < heatmap.shape[0] - 1:
                        val += [heatmap[y + 1, x + 1]]

                if y > 0:
                    val += [heatmap[y - 1, x    ]]
                if y < heatmap.shape[0] - 1:
                    val += [heatmap[y + 1, x    ]]

                # that's the center:
                val += [heatmap[y    , x    ]]

                if sum(val)/len(val) < 5/len(val):
                    heatmap_noise_removed[y, x] = 0

        # for the next pass
        heatmap = heatmap_noise_removed

    return heatmap_noise_removed


def show(img):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    img = cv2.imread('training_data/test_2.jpg')

    start = time.time()

    heatmap = random_heatmap(img)
    # heatmap = stride_heatmap(img)
    
    end = time.time()
    print(round(end - start, 3), 'Seconds')

    h, w, _ = img.shape
    show((cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)/2+0.5) * img.mean(axis=2))


if __name__ == '__main__':
    main()
