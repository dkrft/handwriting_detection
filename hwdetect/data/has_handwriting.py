#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
This module is used to identify the presence of handwritten text in an image
(or sub-sample of an image) using a (sub-)mask. Two different criteria are
possible for this classification:
    * inset box: used to see if mask contains handwritten element in a
      centered box of a specified size; chosen with args.noSide or
      args.noGridSide = False

    * number of non-white pixels: masks image and sees within remaining area
    how many non-white pixels there are (indicating, hopefully, a handwritten
    element); chosen with args.noSide or args.noGridSide = True

This classification is done with labeler(). To run this for various samples
(particular for the sampler scripts), hw_tester() is used.

Usage
-------
>>> import cv2
>>> from hwdetect.data.has_handwriting import hw_tester
>>> from hwdetect.data.defaults import get_default_parser()

>>> parser = get_default_parser()

To view defined keys in argparser:
>>> parser.print_help()

# To modify optional parser objects
>>> parser.grid = True

To pass the required information (number of samples per page and path to HDF);
these values are not needed for this example as we're only considering one
image, so enter whatever you'd like (just need an int and str).Ttypically these
functions would be called by loops iterating through many images specified in
an HDF file
>>> args = parser.parse_args(['1', ".."])

>>> image = cv2.imread("img/page00001.jpg")
>>> mask = cv2.imread("text_mask/page00001.jpg")

>>> labels = hw_tester(args, image, mask)

"""

from collections import Counter
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .defaults import index

__author__ = "Ariel Bridgeman"
__version__ = "1.0"

# need to convert to grayscale and invert before combine with a mask
gray_image = lambda image: cv2.bitwise_not(cv2.cvtColor(image,
                                                        cv2.COLOR_BGR2GRAY))
# mask grayscaled image
mask_image = lambda image, mask: cv2.bitwise_and(gray_image(image),
                                                 gray_image(image), mask=mask)


def plot_image_masks(args, image, mask, label):
    """
    Plot image, mask, and based other useful images based on the following
    criteria:
        * args.noSide == False
            plots inset box (criterion for has handwriting) in image and mask

        * args.noSide == True
            criterion for handwriting based on number of non-white pixels
            in masked image; this is the rudimentary test of whether
            a handwritten mark is present in the masked image. thus, also
            plots gray-scale image and masked image.

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)
    image : list of int
        pixel map of image
    mask : list of int
        pixel map of mask
    label: int
        1, 0 indicating whether handwriting is present or not

    """
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    # need to covert to RGB as CV2 default is BGR
    # matplotlib in RGB
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if args.noSide:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        ax3 = axes[1, 0]
        ax4 = axes[1, 1]
    else:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        ax1 = axes[0]
        ax2 = axes[1]

        side = args.side if not args.grid else args.gridSide
        size = args.box if not args.grid else args.gridBox
        width = size - 2 * side

        # line width for inset rectangle
        lw = 2 if not args.grid else 1
        rgb_img = cv2.rectangle(rgb_img, (side, side),
                                (side + width, side + width),
                                (160, 101, 179), lw)
        mask = cv2.rectangle(mask, (side, side), (side + width, side + width),
                             (160, 101, 179), lw)

    fig.suptitle("hasHW=%i" % label, fontsize=16)

    ax1.set_title('Image', fontsize=15)
    ax1.imshow(rgb_img, vmin=0, vmax=255, origin="upper", aspect='equal')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_title('Mask', fontsize=15)
    ax2.imshow(mask, vmin=0, vmax=255, origin="upper", aspect='equal',
               cmap="gray")
    ax2.set_xticks([])
    ax2.set_yticks([])

    if args.noSide:
        ax3.set_title('Grayscaled image', fontsize=15)
        ax3.imshow(gray_image(image), vmin=0, vmax=255, origin="upper",
                   aspect='equal')
        ax3.set_xticks([])
        ax3.set_yticks([])

        ax4.set_title('Masked grayscale', fontsize=15)
        ax4.imshow(mask_image(image, mask), vmin=0, vmax=255, origin="upper",
                   aspect='equal')
        ax4.set_xticks([])
        ax4.set_yticks([])

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()


def plot_image_grid(args, image, mask, labels):
    """
    Plot image and mask with labeled grid (heat_map)

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)
    image : list of int
        pixel map of image
    mask : list of int
        pixel map of mask
    labels: list of int
        1s, 0s indicating whether handwriting is present or not
    """
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    div = args.box // args.gridBox
    hasHW = np.array(labels)
    hasHW = hasHW.reshape(div, div)

    extent = 0, args.box, 0, args.box

    colors = ((0, 1, 1), (1, 1, 0))
    cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.set_title('Image', fontsize=15)
    ax1.imshow(rgb_img, vmin=0, vmax=255, origin="lower", aspect='equal')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_title('Image with hasHW', fontsize=15)
    ax2.imshow(rgb_img, vmin=0, vmax=255, origin="upper", aspect='equal')
    ax2.imshow(hasHW, vmin=0, vmax=1, alpha=0.5, cmap=cmap,
               interpolation='nearest', extent=extent, aspect="equal")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.set_title('Mask', fontsize=15)
    ax3.imshow(mask, vmin=0, vmax=255, origin="upper", aspect='equal',
               cmap="gray")
    keep = ax3.imshow(hasHW, vmin=0, vmax=1, alpha=0.5, cmap=cmap,
                      interpolation='nearest', extent=extent, aspect="equal")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Set the colorbar labels and title
    colorbar = plt.colorbar(keep, ax=axes.ravel().tolist())
    colorbar.set_label('hasHW', rotation=270)
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])

    # hack to draw gridlines
    for _in in index(args.box, args.gridBox):
        ax1.axvline(x=_in, ls='-')
        ax1.axhline(y=_in, ls='-')
        ax2.axvline(x=_in, ls='-')
        ax2.axhline(y=_in, ls='-')
        ax3.axvline(x=_in, ls='-')
        ax3.axhline(y=_in, ls='-')

    plt.show()


def count_nonwhite(image, mask, white_thresh=15):
    """
    Count number of non-white pixels in a masked image

    If sub_mask has black pixels, it indicates the prescence of a
    handwritten element. As masks may be overdrawn, the mask is applied to
    a grayscaled image. The number of pixels above a threshold
    are summed.

    Parameters
    ----------
    image: list of int
        pixel map of image
    mask : list of int
        pixel map of mask
    white_thresh: int (0, 255)
        integer to give threshold above which counts as a non-white pixel

    Returns
    ----------
    int
        count of pixels with a grayscale value greater than 15 (near white)
    """

    # mask image and count number of unique grayscale values
    mask_img = mask_image(image, mask)
    mvals_dic = Counter(mask_img.flatten())

    # select items with a grayscale value above white threshold
    mvals = [item for key, item in mvals_dic.items() if key > white_thresh]

    return sum(mvals)


def labeler(args, image, mask, num_thresh=10):
    """
    Label an image as having handwriting (1) or not (0)

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)
    image : nested list of int
        pixel map of image
    mask : nested list of int
        pixel map of mask
    num_thresh: int
        threshold for the number of non-white pixels needed to say to
        indicate the handwriting is present in a masked image

    Returns
    ----------
    int
        0 or 1 to indicate lack or presence of handwriting
    """

    # counts number of pixels in mask associated with handwritten elements
    num_hw_pixels = np.sum(mask.flatten() == 255)

    # using number of whites to determine if HW still present
    if args.noSide:
        num_non_whites = count_nonwhite(image, mask)
        if num_hw_pixels > 0 and num_non_whites > num_thresh:
            return 1
        else:
            return 0

    # using default with inset box evaluation of the mask for HW elements
    else:
        side = args.side if not args.grid else args.gridSide
        box = args.box if not args.grid else args.gridBox

        # sum number of pixels with HW inset of sample's mask
        # does NOT guarantee that HW there, as mask may be overdrawn
        # could also be used to provide context for CNN
        inset_mask = mask[side:box - side, side:box - side]
        inset_sum = np.sum(inset_mask.flatten() == 255)
        if num_hw_pixels > 0 and inset_sum > 0:
            return 1
        else:
            return 0


def hw_tester(args, image, mask, num_thresh=10):
    """
    Test existence of handwritten text in a box or grid (if args.grid=True);
    labels the sample or (args.gridBox**2 sub-samples)

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)
    image : nested list of int
        pixel map of image
    mask : nested list of int
        pixel map of mask
    num_thresh : int
        threshold for the number of non-white pixels needed to say to
        indicate the handwriting is present in a masked image

    Returns
    ----------
    labels : list of int
        1 if handwriting present, else 0; returns list of labels for grid
    """

    # labeling sample in a grid layout
    if args.grid:
        labels = []
        # constructing list of indices for grid cells
        indices = index(args.box, args.gridBox)

        for ys in indices:
            for xs in indices:
                # select grid cell of image and mask
                cell_image = image[ys:ys + args.gridBox, xs:xs + args.gridBox]
                cell_mask = mask[ys:ys + args.gridBox, xs:xs + args.gridBox]

                label = labeler(args, cell_image, cell_mask,
                                num_thresh=num_thresh)
                labels.append(label)

                if (args.debug and args.showCells and label == 1) or \
                        (args.showAll and args.showCells):
                    plot_image_masks(args, cell_image, cell_mask, label)

        # plots image and mask with grid overlay
        if (args.debug and sum(labels) > 0) or (args.showAll):
            plot_image_grid(args, image, mask, labels)

    # just 1 label per sample
    else:
        label = labeler(args, image, mask, num_thresh=num_thresh)
        labels = [label]
        if (args.debug and label == 1) or (args.showAll):
            plot_image_masks(args, image, mask, label)

    return labels
