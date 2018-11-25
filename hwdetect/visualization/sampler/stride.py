import numpy as np
from .sampler import Sampler
from hwdetect.utils import show
from random import uniform

class Stride(Sampler):

    def __init__(self, stride=None, y_random=0.25):
        """samples chunks out of the image using a sliding window

        Parameters
        ----------
        stride : int, optional
            by how much to slide the window in x and y direction each step.
            If None, will take 1/3 of the predictors sample size.
        y_random : float, optional
            percent of the predictors sample size to randomly vary in y
            direction between - and + y_random. Default: 0.1
        """
        
        self.stride = stride
        self.y_random = y_random


    def sample(self, image, predictor, label_aggregator):
        """Draw samples from the specified image and predict their labels.

        Parameters
        ----------
        image : np.array
            The image from which the samples are drawn
        predictor : neural_network.predictor.Predictor
            The predictor object that is used for predicting the labels of a sample.
        label_aggregator : function from list of floats to float
            The function that is used for combining the labels predicted by the predictor into a single label.

        Returns
        -------
        dict mapping tuple of int to float
            A dictionary containing the sampled predictions. The keys of the dictionary correspond to the positions of
            the samples while values of the dictionary correspond to the predicted label. The positions is encoded as a
            tuple of integers such that the first integer denotes the y-coordinate and the second integer denotes the
            x-coordinate.
        """

        # get size of image
        height = image.shape[0]
        width = image.shape[1]
        stride = self.stride
        sample_size = predictor.get_image_size()

        if stride is None:
            stride = sample_size//3
            
        # pad image
        """# shorthands:
        c = sample_size # chunk size
        w = width
        h = height
        s = self.stride

        # explanation of the formula:
        # 1.) the stride can be executed w//s times until the left edge of the chunk would be !outside! of the image
        # when that left edge index is the same as the last index of the image + 1, that case will be covered later (A)
        # 2.) no see how many px that actually are by multiplying with the stride
        # 3.) see what the difference to w is, that means, how many px from the right is the left edge of the chunk
        # 4.) look at the difference to c, which is how many px of the chunk would be missing in the image. that's the padding
        # 5.) now for (A). It might be that this value is equal to c, so modulo with c to get 0 which means it fits perfectly
        border_h = (c - (w - (w//s)*s)) % c
        border_v = (c - (h - (h//s)*s)) % c"""

        # well, or you just add the chunk size onto it which would prevent stepping over it
        # for good with 100% more easy to debug code tadaa with a small extra memory overhead

        padded_img = np.pad(image, ((0, sample_size), (0, sample_size), (0, 0)), 'edge')

        X = range(0, width-sample_size, stride)
        Y = range(0, height-sample_size, stride)

        total_num_predictions = len(X) * len(Y)

        # make predictions
        progress = 0
        predictions = {}
        for x in X:
            for y in Y:

                # print approx every 10 percent
                if progress % np.ceil(total_num_predictions/(100/10)) == 0:
                    print("{}% of prediction complete".format(int(progress/total_num_predictions*100)))

                y += int(sample_size * uniform(-self.y_random, self.y_random))
                y = max(0, min(height, y))

                #[None, :] adds another axis to the left of the shape: (1, ...)
                # -> wraps the whole thing into square brackets
                chunk = padded_img[y:y + sample_size, x:x + sample_size][None, :]

                if chunk[0].mean() < 250:
                    prediction = predictor.predict(chunk)[0]
                    predictions[(y, x)] = label_aggregator(prediction)
                    # print(prediction)
                    # show(chunk[0])
                else:
                    predictions[(y, x)] = 0

                progress += 1

        return predictions
