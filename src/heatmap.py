import numpy as np
import cv2
from matplotlib import pyplot as plt
from random import uniform
from preprocessor import Handwriting_Preprocessor
from classifier import HWInterface


def classify_chunk(chunk, predictor):
    """
    parameter
    ---------
    chunk : uint8 numpy.ndarray of shape (150, 150, 3)

    returns
    -------
    [probability of class "handwriting"]
    
    """

    return [predictor.predict([chunk])]


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

    # img = preprocess(img)

    # show(img)

    # divide img into chunks
    stride = 10
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

    predictor = HWInterface()

    print('predictor loaded.')

    y_pos = 0
    while y_pos < hm_h:

        x_pos = 0
        while x_pos < hm_w:

            a = y_pos*stride
            b = x_pos*stride
            chunk = img[a:h+a, b:w+b]
            # don't send white into the classifier
            # to improve performance
            c = 0
            if chunk.min() < chunk.max() - 25:
                # [0] because it classifies multiple images
                # another [0] because there is only one class
                c = classify_chunk(chunk, predictor)[0][0]
                # show(chunk)
            heatmap[y_pos, x_pos] = c
            
            x_pos += 1

        # show 10% steps:
        if y_pos % int(hm_h/10) == 0:
            print(y_pos+1, '/', hm_h, 'done')

        y_pos += 1

    return heatmap


def show(img):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


img = cv2.imread('../data/raw/page0002.jpg')
heatmap = generate_heatmap(img)

print(img.shape)
show(img)

print(heatmap.shape)
show(heatmap)
