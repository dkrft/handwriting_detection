"""
Used to sample pages listed in pandas dataframe. Intended for usage in the
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
                        default="../../data", help="path to directory of \
                        original images and masks")

    parser.add_argument("--saveDir", dest='saveDir', type=str, default="./",
                        help="path to directory to save sampling files")
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

    # hdf of saved data
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
    new_path = args.imgDir + path.split('data')[1]

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
            new_mask_path = args.imgDir + mask_path.split('data')[1]
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


def equalizer(args, date, HW, num_HW, noHW, num_noHW):
    """
    Equally select HW and noHW elements using num_HW
    as sample size for each population

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)

    date: str
        date taken from path of HDF file containig tabulated data

    HW : str
        name of pickled file with HW elements from main()

    num_HW : int
        number of added HW elements; used for loading pkl file

    noHW : str
        name of pickled file with noHW elements from main()

    num_noHW : int
        number of added noHW elements; used for loading pkl file

    Outputs
    ----------
    equalSamp*.pkl
        pickled file with equal selection from random*.pkl
        of cropped images with or without handwritten text;
        need to randomly select from/shuffle as ordered data

    """
    pixels = []
    labs = []
    f = open(HW, "rb")
    for it in range(1, num_HW):
        print(it, num_HW)
        pixels.extend(pkl.load(f))
        labs.extend(pkl.load(f))

    # how many samples of HW and noHW we want
    limit = len(pixels)
    labels = np.array(labs)
    pixels = np.array(pixels)

    # performing noHW selection in 5 groups with remainders ignored
    # used to reduce load on memory
    jt = 1
    obj = []
    obj_labs = []
    g = open(noHW, "rb")
    group_size = num_noHW // 5
    sel_size = limit // 5

    for page in range(1, num_noHW):
        if jt < group_size and jt != num_noHW:
            obj.extend(pkl.load(g))
            obj_labs.extend(pkl.load(g))
            jt += 1

        if jt == group_size:
            print("Processing %s objects; selecting %s" %
                  (len(obj), sel_size))

            # creatig numpy arrays for mask selection
            np_objs = np.array(obj).copy()
            np_objs_lab = np.array(obj_labs).copy()

            rand_sel = np.random.randint(0, high=len(obj) - 1,
                                         size=sel_size)
            # selecting noHW objects
            sel_noHW = np_objs[rand_sel]
            sel_labs = np_objs_lab[rand_sel]

            # appending to HW elems
            pixels = np.append(pixels, sel_noHW, axis=0)
            labels = np.append(labels, sel_labs, axis=0)

            # starting over
            jt = 1
            obj = []
            obj_labs = []

    if len(pixels) == len(labels) and np.abs(len(pixels) - 2 * limit) < 4:
        name = "%s/equalSamp_%s_%s.pkl" % (args.saveDir, date, limit)
        h = open(name, 'wb')
        pkl.dump(pixels, h)
        pkl.dump(labels, h)
        h.close()
        print("Equalizer succeeded! File %s" % name)
    else:
        print("ERROR in equalizer()")

    f.close()
    g.close()


def main(args):
    """
    Execute the multiprocessing of mp_sampler() and equalizer()

    Parameters
    ----------
    args: argparser object specified in terminal command

    Returns
    ----------
    random*.pkl:    pickled file with all the random samples;
                    for ea. page, save list with all pixel maps & list
                    with all labels

    equalSamp*.pkl: pickled file with equal selection from random*.pkl
                    of cropped images with or without handwritten text;
                    need to randomly select from/shuffle as ordered data

    """

    # reading in HDF of all labeled elements
    hdf_path = args.input[0]
    data = pd.read_hdf(hdf_path)

    # ensuring save directory exists
    if not os.path.isdir(args.saveDir):
        os.mkdir(args.saveDir)

    # creating files to save out random selections
    base = os.path.splitext(os.path.basename(hdf_path))[0]
    filebase = "%s/random_%s_samp%s_box%s_side%s" % (args.saveDir, base,
                                                     args.samples[0],
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
    pool = mp.Pool(mp.cpu_count() - 1)
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
            pkl.dump(dic["noHW_img"], g_noHW)
            pkl.dump(dic["noHW_lab"], g_noHW)
            num_pages += 1

        if len(dic["HW_img"]) > 0:
            pkl.dump(dic["HW_img"], f_HW)
            pkl.dump(dic["HW_lab"], f_HW)
            num_pagesHW += 1

        if count % 10 == 0:
            print("%s pages processed" % count)

    # after running randomizer; re-process data to get roughly
    # equal number of HW and no HW n x n px images
    equalizer(args, base, HW_file, num_pagesHW, noHW_file, num_pages)


if __name__ == '__main__':
    parser = get_default_parser()
    main(parser.parse_args())
