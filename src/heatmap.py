import numpy as np
import cv2
from matplotlib import pyplot as plt
from random import uniform
from preprocessor import Handwriting_Preprocessor
from classifier import HWInterface
import time


def classify_chunk(chunk, predictor):
    """
    parameter
    ---------
    chunk : uint8 numpy.ndarray of shape (150, 150, 3)

    returns
    -------
    [probability of class "handwriting"]
    
    """

    return [predictor.predict(chunk)]


def preprocess(img):
    """
    parameter
    ---------
    img : uint8 numpy.ndarray
        2D numpy array of floats (BW) or of 3-tuples of floats (RGB).
        Scanned document that might or might not contain handwriting

    """
    preproc = Handwriting_Preprocessor()
    output = preproc.detect_handwriting(img)
    return output


def generate_heatmap(img):
    """
    Takes raw image, returns heatmap.

    parameter
    ---------
    input : uint8 numpy.ndarray
        2D numpy array of floats (BW) or of 3-tuples of floats (RGB).
        Scanned document that might or might not contain handwriting

    """

    img = preprocess(img)

    # show(img)

    # divide img into chunks
    stride = 50

    # the network has 150x150
    # input neurons
    w = 150
    h = 150

    # add padding to img if wanted
    padding = 0 # TODO will modify img

    # hm = heatmap
    hm_w = int((img.shape[1] - w) / stride) + 1
    hm_h = int((img.shape[0] - h) / stride) + 1

    # Classify each chunk and write
    # results into heatmap.
    # Note, that the img objects are of shape (height, width)
    # opencv resizes and such however use (width, height) for their parameters and stuff
    # (no need to transpose, but it means just don't use img.shape on opencv params)
    heatmap = np.zeros((hm_h, hm_w))

    predictor = HWInterface(1000)

    print('CNN loaded')

    y_pos = 0
    while y_pos < hm_h:

        x_pos = 0
        while x_pos < hm_w:

            a = y_pos*stride
            b = x_pos*stride

            chunk = img[a:h+a, b:w+b]

            # skip everything that has too few dark pixels
            # if (chunk < chunk.mean()).astype(int).sum() / (150*150) > 0.015:
            # if chunk.min() < (chunk.max() - 150):
            if (255 - chunk).sum() > 255 * 500:

                c = classify_chunk([chunk], predictor)
                # c[0] because there is only one class
                # c[0][0] because there is only one chunk per classification
                heatmap[y_pos, x_pos] = c[0][0]

            x_pos += 1

        # show 10% steps:
        if y_pos % int(hm_h/10) == 0:
            print(y_pos+1, '/', hm_h, 'done')

        y_pos += 1

    heatmap = postprocess(heatmap)
    heatmap = postprocess(heatmap)

    return heatmap


def postprocess(heatmap):
    """Removes noise"""
    
    heatmap_noise_removed = heatmap.copy()

    for x in range(1, heatmap.shape[1]-1):
        for y in range(1, heatmap.shape[0]-1):
            
            # the 7 surrounding pixels including itself
            # need to exceed a sum of 4, otherwise it's
            # considered noise.
            val = 0

            val += heatmap[y-1, x-1]
            val += heatmap[y-1, x-0]
            val += heatmap[y-1, x+1]

            val += heatmap[y-0, x-1]
            val += heatmap[y-0, x-0] # that's the center
            val += heatmap[y-0, x+1]

            val += heatmap[y+1, x-1]
            val += heatmap[y+1, x-0]
            val += heatmap[y+1, x+1]

            if val < 5:
                heatmap_noise_removed[y, x] = 0
                
    return heatmap_noise_removed


def show(img):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


img = cv2.imread('../data/raw/page0001.jpg')

start = time.time()
heatmap = generate_heatmap(img)
end = time.time()
print(round(end - start, 3), 'Seconds')

print(img.shape)
show(img)

print(heatmap.shape)
show(heatmap)
