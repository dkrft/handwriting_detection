from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from scipy import signal
from hwdetect.utils import show
from scipy import ndimage
from hwdetect.preprocessor.preprocessor import Preprocessor

class Bandpass(Preprocessor):

    def __init__(self, noise_x=15, noise_y=3, noise_t=15,
                 grow_x=14, grow_y=8, order=5,
                 t=2.1, m=70, w=1000,
                 verbose=False):
        """img is a numpy 2D array of dtype uint8 (greyscale image).
        Returns the image with some detected machine writings removed.
        
        if too many false positives (= handwriting removed from img), try
        to reduce t and increase noise_t.
        
        looks for horizontal frequencies and does a butterworth
        bandpass between frequencies of 0.25 to 0.75, which makes
        machine writing pop out, since it has a very regular pattern.

        then thresholds this bandpassed image, such that True values
        are scattered across the machine writing and False values on the
        rest of the picture.

        There will also be some sparse noise. The cleaner and regular handwriting is,
        the more false classifications will be on handwriting; that is, the more
        similar the handwriting is to machine writing.

        When handwriting, even a very clean one, is in an angle, it will be more likely ignored by this method.

        Then noise is removed and the mask is grown such that it covers
        the machine writing completely. Then the machine writing can
        be removed."""
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.noise_t = noise_t
        self.grow_x = grow_x
        self.grow_y = grow_y
        self.order = order
        self.t = t
        self.m = m
        self.w = w


    def grow(self, mask, px_x=10, px_y=5):
        """
        simplified demonstration:
            [False, False, True, False, False]
        becomes:
            [False, True, True, True, False]
        the mask was grown by 1px

        pros: it's my custom style mask growing c:,
        it captures the shape of the text quite well

        cons: it's the slowest
        """
        # faster aproach to growing the mask because it uses
        # numpy-array-masking instead of looping over every px in native
        # python. X and Y are arrays containing horizontal and vertical
        # indices and that only contain those indices, that correspond to
        # a 1 in mask. Then it is iterated over X and Y and grown by over-
        # writing neighboring pixels with 1. Since I iterate over X and Y,
        # I get the ositions for all the positions in the mask that have
        # to grow.

        assert px_x >= 1 and px_y >= 1
        # otherwise will consume gigabytes of ram:
        assert mask.dtype == bool
        mask2 = mask.copy()

        Y, X = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        Y = Y[mask].flat
        X = X[mask].flat

        for x, y in zip(X, Y):
            t = max(y-px_y, 0)
            b = min(y+px_y, mask.shape[0]-1)
            l = max(x-px_x, 0)
            r = min(x+px_x, mask.shape[1]-1)
            mask2[t:b, l:r] = True

        return mask2


    def fast_grow(self, mask, a=10, b=5):
        """
        chaining some cv2 resizes to grow the mask. First downscale,
        cast to bool (which makes anything antialiased to True), then scale
        up, which makes it super blury, cast to bool again which makes the blur
        to True values.

        pros: the fastest implementation here for growing only
        """

        h, w = mask.shape
        out = cv2.resize(mask.astype(float), (w//a, h//b), interpolation=cv2.INTER_AREA).astype(bool)
        out = cv2.resize(out.astype(float), (w, h), interpolation=cv2.INTER_LINEAR).astype(bool)
        return out


    def fast_grow_and_noise_removal(self, img, px_x=10, px_y=5, noise_threshold=0.04):
        """
        removes noise and grows the mask by applying a blur
        filter, and then thresholding the result.
        noise_threshold should be > 0 to produce useful results.

        pros: also removes noise. It's the fastest
        way to do both.
        
        cons: on the start and end of machine written lines,
        some px are not captured
        """

        # 1.: grow in both directions, hence *2
        # 2.: but make *3 because it tends to select less than the other methods
        # box blur kernel
        kernel = np.ones((px_y*3, px_x*3))
        kernel /= kernel.sum()

        # apply kernel
        img_blur = cv2.filter2D(img.astype(float), -1, kernel)

        # create mask from blurred image
        # increase noise_threshold to remove more noise
        img_blur[img_blur < noise_threshold] = 0
        img_blur = img_blur.astype(bool)
        
        return img_blur


    def remove_noise(self, mask, px_x=15, px_y=3, threshold=15):
        """
        counts True values in the rectangle between -px_x,
        px_x, -px_y and px_y, and if there are less True
        values than the threshold, the value is set to False.
        """

        # similar to grow.

        assert px_x >= 1 and px_y >= 1
        # otherwise will consume gigabytes of ram:
        assert mask.dtype == bool
        mask2 = mask.copy()

        Y, X = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        Y = Y[mask].flat
        X = X[mask].flat

        for x, y in zip(X, Y):
            t = max(y-px_y, 0)
            b = min(y+px_y, mask.shape[0]-1)
            l = max(x-px_x, 0)
            r = min(x+px_x, mask.shape[1]-1)
            if mask[t:b, l:r].sum() <= threshold:
                mask2[y, x] = False

        return mask2



    def bandpass_image(self, img, m, t, verbose=False):

        # https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

        # 2. do bandpass on the whole flattened image,
        # so that it simply is an 1D array/signal
        signal = img.flatten()

        # cut frequencies of 0.25hz - 0.75hz
        # those numbers were optimized by hand 
        lowcut = 0.25
        highcut = 0.75

        order = 5
        b, a = butter(order, [lowcut, highcut], btype='band')
        filtered = lfilter(b, a, signal)

        # signal was flattened. now reshape to the image shape
        filtered = filtered.reshape((img.shape[0], img.shape[1]))

        # the filtered image has a shift to the right. move it back by cutting left and adding right
        shift = 4 # positive px to the left
        # print(shift, filtered.shape, type(shift))
        filtered[:,shift:]
        filtered = np.concatenate([filtered[:,shift:], np.zeros((len(filtered), shift))], axis=1)

        if verbose:
            show(filtered)

        # 3. create mask
        
        # set threshold such that the machine writing becomes
        # highlighted, but the handwriting not.
        # example for filtered.min() & filtered.max(): -141.9 & 141.8
        # 3 seems to work quite good. 2.5, 2.0, 1.5 for less false positives
        # (positive = machine-written)
        # don't go closer to 0 than (e.g. m=70) -70, 70
        lower_threshold = min(-m, filtered.min()/t)
        upper_threshold = max( m, filtered.max()/t)
        # the min and max values are quite individual for each picture,
        # so a fixed threshold does not work, which is the reason why
        # chunking the picture is not a good idea. When the chunk only
        # contains handwriting the threshold will become low automatically
        # and false predictions appear.

        return filtered, lower_threshold, upper_threshold
        

    def preprocess(self, img):

        noise_x = self.noise_x
        noise_y = self.noise_y
        noise_t = self.noise_t
        grow_x = self.grow_x
        grow_y = self.grow_y
        order = self.order
        t = self.t
        m = self.m
        w = self.w

        # for debugging:
        verbose = False

        if verbose:
            start = time.time()


        # 1. prepare img

        # copy for later
        img_masked = img.copy()


        # grayscale. Use min so that blue text becomes black
        if len(img.shape) == 3:
            img = img.min(axis=2)

        # to make it more stable, scale all images to equal width
        # (doing that increases the quality considerably)
        h = int(img.shape[0]*(w/img.shape[1]))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(float)

        # make sure that between 0 and 255
        img -= img.min()
        img /= img.max()
        img *= 255
        # the white paper, which makes for the majority of pixels, should be 0
        # so that be frequency stuff does not get confused by the DC
        img = img.max() - img
        img = img.astype(np.uint8)


        # try to find frequencies:
        filtered, lower_threshold, upper_threshold = self.bandpass_image(img, m, t)


        # print(lower_threshold, upper_threshold)

        
        # restore original rotation
        # filtered = rotate(filtered, -90)

        # now create everywhere based on thresholding
        # the bandpass filtered image. Values close to 0
        # are False, otherwise True. (filtered gives positive
        # and negative values) 
        mask = np.zeros(filtered.shape, np.uint8)
        mask[filtered < lower_threshold] = 1
        mask[filtered > upper_threshold] = 1
        mask = mask.astype(bool)


        # grow the mask so that it covers the text

        """s = time.time()
        # slowest:
        a = remove_noise(mask, noise_x, noise_y, noise_t)
        b = grow(a, grow_x, grow_y)
        print(time.time()-s)

        s = time.time()
        a = remove_noise(mask, noise_x, noise_y, noise_t)
        b = fast_grow(a, grow_x, grow_y)
        print(time.time()-s)

        # fastest:
        s = time.time()
        mask = fast_grow_and_noise_removal(mask, grow_x, grow_y)
        print(time.time()-s)"""

        # decided for that one. It's the slowest but has the best results:
        a = self.remove_noise(mask, noise_x, noise_y, noise_t)
        mask = self.grow(a, grow_x, grow_y)

        # then, scale back to original size
        h, w = img_masked.shape[:2]
        mask = cv2.resize(mask.astype(float), (w, h), cv2.INTER_NEAREST).astype(bool)


        # 4. plot if wanted
        # note, that img_masked is not masked at this point
        # but rather a copy of the untouched original (except)
        # that it might be transposed
        if verbose:
            end = time.time()
            print(round(end-start, 3), 'Seconds')
            # show(filtered)
            # show(img/5 + (mask_not_grown+1)*128)
            if len(img_masked.shape) == 3:
                # RGB to Grayscale using mean
                show(img_masked.mean(axis=2) + mask*255)
            else:
                show(img_masked + mask*255)



        # 5. apply mask

        if len(img_masked.shape) == 3:
            img_masked[:,:,0][mask] = 255
            img_masked[:,:,1][mask] = 255
            img_masked[:,:,2][mask] = 255
        else:
            img_masked[mask] = 255


        return img_masked
        