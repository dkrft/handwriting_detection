"""
Sample pages listed in pandas dataframe. Intended for usage in the
terminal like:

python sample_pages.py 250 ./labels/26-10.hdf

OR

python sample_pages.py 250 ./labels/26-10.hdf --box 100 --side 20

To use in script directly:
>>> import sample_pages_tensor2 as samp
>>> parser = samp.get_default_parser()
>>> vals = parser.parse_args(['1', "./labels/26-10.hdf"])

# e.g. to modify other parser objects
>>> parser.grid = True

>>> samp.main(vals)

"""


import argparse
from collections import Counter, defaultdict
import cv2
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle as pkl

items = lambda arg, df: [arg] * len(df)


def get_default_parser():
    """
    Obtain default parser for script

    Returns
    -------
    argparse.ArgumentParser
        argument parser object

    """

    # create parser object
    parser = argparse.ArgumentParser(description='Sample pages for CNN use.')

    # requires a number of samples per page and HDF files with pages tabulated
    parser.add_argument('samples', metavar='S', type=int, nargs=1,
                        help='number of desired samples per page')
    parser.add_argument('input', metavar='filename', type=str, nargs=1,
                        help='name of full HDF (not hasHW)')

    # assumes that soft-linked data directory is +1 outside of full git dir
    # so +2 ../../ outside of hwdetect/data
    parser.add_argument("--imgDir", dest='imgDir', type=str,
                        default="../../data/original_data/", help="path to \
                        directory of original images and masks")

    # where and how to save samples & labels
    parser.add_argument("--saveDir", dest='saveDir', type=str,
                        default="../../data/training_data/",
                        help="path to directory to save sampling files")
    parser.add_argument("--saveAsDict", dest='saveAsDict', default=False,
                        action='store_true', help="save as dictionary in \
                        pkl file (default class)")

    # simple specifications for samples & debug mode
    parser.add_argument('--box', dest='box', type=int,
                        default=150, help="int n (150) for creating n x n pixel \
                        box")
    parser.add_argument('--debug', dest="debug", default=False,
                        action='store_true', help="use matplotlib to display \
                        potential HW elements no longer parallel processing")

    # labels given if handwriting is present in only a central box of mask
    # otherwise checks for non-white pixels present above a threshold
    parser.add_argument('--noSide', dest='noSide', default=False,
                        action='store_true', help='do not evaluate sub-, \
                        centered box within box as having handwriting or not; \
                        paired with --side')
    parser.add_argument('--side', dest='side', type=int, default=20,
                        help='int s (20) for creating nested (n-2s)x(n-2s) box')

    # choose to sub-sample n**2 box in a grid of hwbox**2 boxes
    parser.add_argument('--grid', dest='grid', default=False,
                        action='store_true', help="mark sub-samples of box as \
                        having handwritten elements or not")
    parser.add_argument('--hwbox', dest='hwbox', type=int,
                        default=15, help="int n (15) for n x n pixel sub-box \
                         for labeling if HW or not; must be int factor of box")

    parser.add_argument("--nproc", dest="nproc", type=int,
                        default=mp.cpu_count() - 1, help="int for number of \
                        processes to simultaneously run (machine - 1)")

    parser.add_argument("--mixFactor", dest="mixFactor", type=float,
                        default=1., help="float to determine how many noHW \
                        elements to select in data_mixer() given the limit\
                        of the number of HW elements")

    # hdf of saved data for stats
    # fast save with just sampling handwriting

    # # selection criteria for samples
    # parser.add_argument('--keep_machSig', dest='machSig', type=bool,
    #                     default=False, help="boolean to keep sampled image \
    #                     with machine signature or not (False)")
    return parser


def hw_tester(args, img, mask):
    """
    Test existence of handwritten text in a box or grid (if args.grid=True);
    labels the sample or (args.hwbox**2) sub-samples of the mask & image
    of random_crop

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)

    img : nested list of int
        pixel map of cropped image from random_crop(); (args.box, args.box, 3)

    mask : nested list of int
        pixel map of cropped mask from random_crop(); (args.box, args.box, 1)

    Returns
    ----------
    labels : list of int
        1 if handwriting present in sub-sample; else 0
    """
    def labeler(sub_img, sub_mask):
        """
        Label sub-images.

        If sub_mask has black pixels, it indicates the prescence of a
        handwritten element. As masks may be overdrawn, the mask is applied to
        a grayscaled sub_image. The number of pixels above a threshold of 15
        (relatively white) are summed, and if they exceed a value, a
        label of 1 is bestowed. If any of the criteria are not met, then the
        sub-sample is given a 0.

        Parameters
        ----------
        sub_img : list of int
            pixel map of (sample of) cropped image from random_crop()

        sub_mask : list of int
            pixel map of (sample of) cropped mask from random_crop()

        Internal parameters
        --------------------
        args.grid : bool
            if True, the args.box**2 sample is sub-divided into args.hwbox**2
            images and masks; these are then scrutinized individually.
            Otherwise, only 1 label is assigned to the entire args.box**2
            sample (img).

        Returns
        ----------
        int
            count of pixels with a grayscale value greater than 15 (near white)

        """

        # need to convert to grayscale before mask
        gray_img = cv2.bitwise_not(cv2.cvtColor(sub_img,
                                                cv2.COLOR_BGR2GRAY))
        # mask grayscale image
        mask_img = cv2.bitwise_and(gray_img, gray_img,
                                   mask=sub_mask)

        # gvals_dic = Counter(gray_img.flatten())
        mvals_dic = Counter(mask_img.flatten())

        # select items with a grayscale value above white threshold
        mvals = [item for key,
                 item in mvals_dic.items() if key > 15]

        if args.debug and sum(mvals) > 0:
            matplotlib.rc('xtick', labelsize=10)
            matplotlib.rc('ytick', labelsize=10)

            fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
            axes[0, 0].set_title('Image', fontsize=15)
            axes[0, 0].imshow(sub_img, vmin=0, vmax=255)

            axes[0, 1].set_title('Mask', fontsize=15)
            axes[0, 1].imshow(sub_mask, vmin=0, vmax=255)

            axes[1, 0].set_title('Grayscaled image', fontsize=15)
            axes[1, 0].imshow(gray_img, vmin=0, vmax=255)

            axes[1, 1].set_title('Masked grayscale', fontsize=15)
            axes[1, 1].imshow(mask_img, vmin=0, vmax=255)

            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            plt.show()

        return sum(mvals)

    if args.grid:

        labels = []
        indices = np.linspace(0, args.box - args.hwbox,
                              (args.box - args.hwbox) / args.hwbox + 1).astype(int)
        for ys in indices:
            for xs in indices:
                # select sub-box of mask
                sub_mask = mask[ys:ys + args.hwbox,
                                xs:xs + args.hwbox]

                # number of pixels with HW mask present
                sub = np.sum(sub_mask.flatten() == 255)

                # select sub-box of image
                sub_img = img[ys:ys + args.hwbox,
                              xs:xs + args.hwbox]

                # checking for HW elements
                mvals = labeler(sub_img, sub_mask)

                if sub > 0 and mvals > 10:
                    labels.append(1)
                else:
                    labels.append(0)
    else:
        # sum number of pixels with HW mask present
        sub = np.sum(mask.flatten() == 255)
        mvals = labeler(img, mask)

        # inset of mask box to ensure HW mask not on edge;
        # does NOT guarantee that HW there, as mask may be overdrawn
        inset_mask = mask[args.side:args.box - args.side,
                          args.side:args.box - args.side]
        inset_sum = np.sum(inset_mask.flatten() == 255)

        if sub > 0 and ((not args.noSide and inset_sum > 0) or
                        (args.noSide and mvals > 10)):
            labels = [1]
        else:
            labels = [0]

    return labels


def random_crop(args, pid, img, mask=[]):
    """
    Create S random selection of a page with a fixed box size

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)

    pid : int
        pageid for saving out img (if specified in args)


    img : nested list of int
        pixel map of cropped image; shape = (args.box, args.box, 3)

    mask : nested list of int
        pixel map of cropped mask; shape = (args.box, args.box, 1)

    Returns
    ----------
    dict
        with cropped images and sub-sample labels sorted into
        noHW_img, noHW_lab, HW_img, HW_lab
    """

    # creating container for holding samples of page
    holder = defaultdict(list)

    samples = 0
    h, w = img.shape[:2]
    num_subs = args.box // args.hwbox
    while samples < args.samples[0]:
        # - bottom left pixel of box -
        rand_h = np.random.randint(0, h - args.box)
        rand_w = np.random.randint(0, w - args.box)
        crop_img = img[rand_h:rand_h + args.box, rand_w:rand_w + args.box]

        if len(mask) == 0:
            holder["noHW_img"].append(crop_img)
            if args.grid:
                holder["noHW_lab"].append([0] * num_subs**2)
            else:
                holder["noHW_lab"].append([0])
        else:
            crop_mask = mask[rand_h:rand_h + args.box, rand_w:rand_w + args.box]

            # mask is within cropped area
            if np.sum(crop_mask.flatten() == 255) / (args.box**2) > 0.03:

                labs = hw_tester(args, crop_img, crop_mask)

                if sum(labs) == 0:
                    holder["noHW_img"].append(crop_img)
                    if args.grid:
                        holder["noHW_lab"].append([0] * num_subs**2)
                    else:
                        holder["noHW_lab"].append([0])
                else:
                    holder["HW_img"].append(crop_img)
                    holder["HW_lab"].append(labs)

            # mask is not in cropped area
            else:
                holder["noHW_img"].append(crop_img)
                if args.grid:
                    holder["noHW_lab"].append([0] * num_subs**2)
                else:
                    holder["noHW_lab"].append([0])

        if args.grid:
            holder["pageid"].append([pid] * num_subs**2)
        else:
            holder["pageid"].append([pid])

        samples += 1

    return holder


def mp_sampler(zipped):
    """
    Load images and masks, rescale, and pass onto random_crop()
    for (args.samples[0]) samplings. Returns result of sampling
    to main.

    Parameters
    ----------
    zipped : zip object
        zipped object of grouped df and argparser object

    Returns
    ----------
    dict
        from random_crop contains samplings of page and
        labels from random_crop()

    """

    # parsing zipped input
    grouped, args = zipped
    _, group = grouped

    # if more than one mask, path will be duplicated
    path = group["path"].unique()[0]
    # as data engineer's relative path may differ from user's
    new_path = args.imgDir + "/".join(path.split(os.sep)[-3:])

    # variable for if saving out random cropped images
    base = (os.path.normpath(new_path)).split(os.sep)[2]
    page_base = os.path.splitext(os.path.basename(new_path))[0]
    pageid = "%s_%s" % (base, page_base)

    # 0 import in image and masks
    img = cv2.imread(new_path)
    h, w = img.shape[:2]

    # 0.a rescale images in way to preserve aspect ratio
    # and help with a more uniform sampling process
    scale_me = 1.
    if h < 2337 and w < 2337:
        if h > w:
            scale_me = 2337 / h
        else:
            scale_me = 2337 / w
    img = cv2.resize(img, (0, 0), fx=scale_me, fy=scale_me)
    h, w = img.shape[:2]

    hasHW = bool(group.hasHW.max())
    # 1.a no masks are present; hasHW = 0
    if not hasHW:
        dic = random_crop(args, pageid, img, mask=[])

    # 1.b has mask(s)
    else:
        or_mask = []
        # 2.a need to load each mask for cropping classification
        for index, el in group.iterrows():
            if el["hwType"] == "mach_sig":
                continue

            # otherwise, handwritten element
            mask_path = el["mask"]
            new_mask_path = args.imgDir + "/".join(mask_path.split(os.sep)[-3:])
            mask = cv2.imread(new_mask_path, 0)
            if len(or_mask) < 1:
                or_mask = mask
            else:
                # combine mark and text masks
                or_mask = cv2.bitwise_or(or_mask, mask)

        # scale mask to be same size of image
        or_mask = cv2.resize(or_mask, (0, 0), fx=scale_me, fy=scale_me)
        dic = random_crop(args, pageid, img, np.array(or_mask))

    return dic


def random_sampler(args):
    """
    Sample randomly from pages listed in dataframe specified by
    args.input[0]

    Parameters
    ----------
    args: argparser object specified in terminal command

    Outputs
    ----------
    random*.txt: text file with # of pickled files

    random*.pkl:    pickled file with all the random samples;
                   for ea. page, save list with all pixel maps & list
                   with all labels

    """

    # reading in HDF of all labeled elements
    hdf_path = args.input[0]
    data = pd.read_hdf(hdf_path)

    # ensuring save directory exists
    if not os.path.isdir(args.saveDir + "/randomSamp/"):
        os.makedirs(args.saveDir + "/randomSamp/")

    # creating files to save out random selections
    base = os.path.splitext(os.path.basename(hdf_path))[0]
    filebase = "%s/randomSamp/rand_%s_samp%s_box%s_side%s" % (args.saveDir, base,
                                                              args.samples[
                                                                  0],
                                                              args.box,
                                                              args.side if not args.noSide else 0)

    HW_file = "%s_HW.pkl" % (filebase)
    noHW_file = "%s_noHW.pkl" % (filebase)

    f_HW = open(HW_file, "wb")
    g_noHW = open(noHW_file, "wb")

    # reducing data to first element to only load img/mask once
    sel = data.groupby(["pageid", "hwType", "path"], as_index=False).first()

    # then grouping by unique page identifiers
    grouped = sel.groupby(["pageid", "path"], as_index=False)

    # save one local node for sanity
    pool = mp.Pool(args.nproc)
    num_pages = 0
    num_pagesHW = 0

    # in debug mode, do not want threading
    if not args.debug:
        iterator = enumerate(pool.imap(mp_sampler,
                                       zip(grouped, items(args, grouped))))
    else:
        iterator = enumerate(zip(grouped, items(args, grouped)))

    for count, result in iterator:

        if not args.debug:
            dic = result
        else:
            dic = mp_sampler(result)

        if len(dic["noHW_lab"]) > 0:
            pkl.dump({"imgs": dic["noHW_img"], "labels": dic["noHW_lab"]},
                     g_noHW)
            num_pages += 1

        if len(dic["HW_img"]) > 0:
            pkl.dump({"imgs": dic["HW_img"], "labels": dic["HW_lab"]},
                     f_HW)
            num_pagesHW += 1

        if count % 10 == 0:
            print("%s pages processed" % count)

    h = open(filebase + ".txt", "w")
    h.write("%s, %s" % (num_pagesHW, num_pages))
    h.close()

    f_HW.close()
    g_noHW.close()


def data_mixer(args):
    """
    Create mixed sample of HW and noHW, where number of HW samples is the
    limiting factor multiplied by the args.mixFactor to determine the number
    of noHW elements

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)

    Outputs
    ----------
    mix*.pkl
        pickled file with specified selection from
        random*.pkl of cropped images with or without handwritten text;
        need to randomly select from/shuffle as ordered data

    """

    hdf_path = args.input[0]
    base = os.path.splitext(os.path.basename(hdf_path))[0]
    filebase = "%s/randomSamp/rand_%s_samp%s_box%s_side%s" % (args.saveDir, base,
                                                              args.samples[0],
                                                              args.box,
                                                              args.side if not args.noSide else 0)

    if not os.path.isdir(args.saveDir + "/mixSamp/"):
        os.makedirs(args.saveDir + "/mixSamp/")

    num_HW, num_noHW = np.loadtxt("%s.txt" % (filebase), delimiter=",",
                                  dtype=int)

    pixels = []
    labs = []
    f = open("%s_HW.pkl" % (filebase), "rb")
    for it in range(num_HW):
        page_dic = pkl.load(f)
        pixels.extend(page_dic["imgs"])
        labs.extend(page_dic["labels"])

    # how many samples of HW and noHW we want
    limit = len(pixels)
    np_labels = np.array(labs)
    np_pixels = np.array(pixels)

    # number of noHW elements to select
    noHW_limit = int(args.mixFactor * limit)

    # performing noHW selection groups with remainders ignored
    # used to reduce load on memory
    pixels_noHW = []
    labs_noHW = []

    # rough way to determine number of groups for memory reduction
    # assume all samples are noHW and takes into account pixel maps
    # divisor determined through trial-and-error with 150 x 150 px image
    mem_safety_factor = (
        num_noHW * args.samples[0] * args.box**2) // (5000 * 150**2)
    group_size = num_noHW // mem_safety_factor
    sel_size = noHW_limit // mem_safety_factor

    g = open("%s_noHW.pkl" % (filebase), "rb")
    jt = 1
    for page in range(1, num_noHW):
        if jt < group_size and jt != num_noHW:
            pass
            page_dic = pkl.load(g)
            pixels_noHW.extend(page_dic["imgs"])
            labs_noHW.extend(page_dic["labels"])
            jt += 1

        if jt == group_size:
            print("Processing %s objects; selecting %s" %
                  (len(pixels_noHW), sel_size))

            # creating numpy arrays for mask selection
            np_pixels_noHW = np.array(pixels_noHW).copy()
            np_labs_noHW = np.array(labs_noHW).copy()

            rand_sel = np.random.randint(0, high=len(pixels_noHW) - 1,
                                         size=sel_size)
            # selecting noHW objects
            sel_noHW = np_pixels_noHW[rand_sel]
            sel_labs = np_labs_noHW[rand_sel]

            # appending to HW elems
            np_pixels = np.append(np_pixels, sel_noHW, axis=0)
            np_labels = np.append(np_labels, sel_labs, axis=0)

            # start new sampling of no_HW elements
            jt = 1
            pixels_noHW = []
            labs_noHW = []

    if len(np_pixels) == len(np_labels):
        name = "%s/mixSamp/mix_%s_HW%s_noHW%s_box%s_side%s.pkl" % (args.saveDir, base,
                                                                   limit,
                                                                   noHW_limit,
                                                                   args.box,
                                                                   args.side if not args.noSide else 0)
        h = open(name, 'wb')
        pkl.dump({"imgs": np_pixels, "labels": np_labels}, h)
        h.close()

        print("\ndata_mixer succeeded! \nFile %s" % name)
    else:
        print("ERROR in data_mixer()")

    f.close()
    g.close()


def main(args):
    """
    Execute the multiprocessing of random_sampler() and data_mixer() in
    sequence, over the specified HDF dataframe

    Parameters
    ----------
    args: argparser object specified in terminal command

    Outputs
    ----------
    rand*.txt: text file with # of pickled files

    rand*.pkl:    pickled file with all the random samples;
                    for ea. page, save list with all pixel maps & list
                    with all labels


    mix*.pkl: pickled file with specified mixed selection from random*.pkl
                    of cropped images with or without handwritten text;
                    need to randomly select from/shuffle as ordered data

    """

    # randomly select X n x n samples from each page listed in the args input
    # HDF dataframe
    # random_sampler(args)
    print()

    # after running randomizer; re-process data to get roughly desired
    # proportion of HW to no HW n x n px images given with args.mixFactor
    data_mixer(args)


if __name__ == '__main__':
    parser = get_default_parser()
    main(parser.parse_args())
