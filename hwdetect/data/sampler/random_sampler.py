#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
This script randomly samples the pages listed in a pandas dataframe.

For successful usage, the dataframe must contain the following columns:
    hwType, hasHW, pageid, path, and mask
with values as described in about_hdf.md. [Files produced by handle_Labelbox.py
and read_PRImA.py already meet these requirements.]

The main function used within this script is:
 * random_sampler() --- samples each page and separately saves images that
    contain or do not contain handwritten elements in pickled dictionaries.
    The dictionaries contain lists of the pixel maps and labels, as well
    as other identifying information.

Intended for usage in the terminal like:

python random_sampler.py 250 ./labeled_databases/26-10.hdf

OR

# specified with optional arguments specified in defaults.get_default_parser()
python random_sampler.py 250 ./labeled_databases/26-10.hdf --box 100 --side 20

To use in script directly:
>>> import hwdetect.data.create_dataset()
>>> from hwdetect.data.defaults import get_default_parser()
>>>
>>> parser = get_default_parser()

To view defined keys in argparser:
>>> parser.print_help()

# To modify optional parser objects for grid mode (not default)
>>> parser.grid = True

To pass the required information (number of samples per page and path to HDF)
>>> args = parser.parse_args(['1', "../../data/labeled_databases/26-10.hdf"])

To run function to randomly select, mix, and separate training and testing data
>>> random_sampler.main(args)

"""

import cv2
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import os
import pickle as pkl
import pandas as pd

from hwdetect.data.defaults import base, get_default_parser, rand_base
from hwdetect.data.has_handwriting import hw_tester

__author__ = "Ariel Bridgeman"
__version__ = "1.0"

# --- convenience functions ---
# generate iterator for multi-processing
items = lambda arg, df: [arg] * len(df)


def random_crop(args, pid, img, mask=[]):
    """
    Create n random selection of a page with a fixed box size

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
    num_subs = args.box // args.gridBox
    while samples < args.samples[0]:
        # - bottom left pixel of box -
        rand_h = np.random.choice(h - args.box, replace=False)
        rand_w = np.random.choice(w - args.box, replace=False)
        crop_img = img[rand_h:rand_h + args.box, rand_w:rand_w + args.box]

        if len(mask) == 0:
            holder["noHW_img"].append(crop_img)
            holder["noHW_page"].append(pid)
            holder["noHW_loc"].append({"x": rand_w, "y": rand_h})
            if args.grid:
                holder["noHW_lab"].append([0] * num_subs**2)
            else:
                holder["noHW_lab"].append([0])
        else:
            crop_mask = mask[rand_h:rand_h + args.box, rand_w:rand_w + args.box]

            # mask is within cropped area
            if np.sum(crop_mask.flatten() == 255) / (args.box**2) > 0.03:

                # checking criteria to see if would consider as having
                # handwriting present or not
                labs = hw_tester(args, crop_img, crop_mask)

                if sum(labs) == 0:
                    holder["noHW_img"].append(crop_img)
                    holder["noHW_page"].append(pid)
                    holder["noHW_loc"].append({"x": rand_w, "y": rand_h})
                    if args.grid:
                        holder["noHW_lab"].append([0] * num_subs**2)
                    else:
                        holder["noHW_lab"].append([0])

                else:
                    holder["HW_img"].append(crop_img)
                    holder["HW_lab"].append(labs)
                    holder["HW_page"].append(pid)
                    holder["HW_loc"].append({"x": rand_w, "y": rand_h})

            # mask is not in cropped area
            else:
                holder["noHW_img"].append(crop_img)
                holder["noHW_page"].append(pid)
                holder["noHW_loc"].append({"x": rand_w, "y": rand_h})
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
        zipped object of grouped df, argparser object, and base of hdf_path

    Returns
    ----------
    dict
        from random_crop contains samplings of page and
        labels from random_crop()

    """

    # parsing zipped input
    grouped, args, baser = zipped
    _, group = grouped

    # if more than one mask, path will be duplicated
    path = group["path"].unique()[0]
    # as data engineer's relative path may differ from user's
    new_path = args.imgDir + "/".join(path.split(os.sep)[-3:])

    # variable for if saving out random cropped images
    page_base = os.path.splitext(os.path.basename(new_path))[0]
    pageid = "%s_%s" % (baser, page_base)

    # 0 import in image and masks
    img = cv2.imread(new_path)

    try:
        h, w = img.shape[:2]

    except:
        print("\nNeed to set imgDir in parser (get_default_parser()). \
            \nPath given in HDF differs from local set-up\
            \nHDF path example: %s" % new_path)
        return "stop"

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
            new_mask_path = args.imgDir + \
                "/".join(mask_path.split(os.sep)[-3:])
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


def random_sampler(args, hdf_path):
    """
    Sample randomly from pages listed in dataframe of hdf_path

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)
    hdf_path : str
        string with full or relative path to an HDF pandas dataframe

    Outputs
    ----------
    random*.txt: text file
        gives number of pickled elements contained in rand*HW.pkl and
        rand*noHW.pkl

    random*.pkl: pickled dictionaries
        contains dictionaries for all random samples created for ea. page,
        dictionary keys includ "imgs" and "labels" which refer to lists with
        all the pixel maps & with all the labels (0 = no handwriting,
        1 = has handwriting), respectively

    """

    # reading in HDF of all labeled elements
    data = pd.read_hdf(hdf_path)

    # ensuring save directory exists
    if not os.path.isdir(args.saveDir + "/randomSamp/"):
        os.makedirs(args.saveDir + "/randomSamp/")

    # creating files to save out random selections
    filebase = rand_base(args, base(hdf_path))
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
                                       zip(grouped, items(args, grouped),
                                           items(base(hdf_path), grouped))))
    else:
        iterator = enumerate(zip(grouped, items(args, grouped)))

    foundData = True
    for count, result in iterator:

        if not args.debug:
            dic = result
        else:
            dic = mp_sampler(result)

        # found in first iteration of mp_sampler that files not present
        # emergency stop
        if dic == "stop":
            foundData = False
            break

        # will not save data unless not in debug mode or specified otherwise
        if not args.debug or (args.debug and args.saveData):
            if "noHW_lab" in dic:
                pkl.dump({"imgs": dic["noHW_img"], "labels": dic["noHW_lab"],
                          "pages": dic["noHW_page"], "locs": dic["noHW_loc"]},
                         g_noHW)
                num_pages += 1

            if "HW_img" in dic:
                pkl.dump({"imgs": dic["HW_img"], "labels": dic["HW_lab"],
                          "pages": dic["HW_page"], "locs": dic["HW_loc"]},
                         f_HW)
                num_pagesHW += 1

        if count % 10 == 0:
            print("%s pages processed" % count)

    if foundData and (not args.debug or (args.debug and args.saveData)):
        h = open(filebase + ".txt", "w")
        h.write("%s, %s" % (num_pagesHW, num_pages))
        h.close()

    f_HW.close()
    g_noHW.close()


if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()
    for hdf in args.inputs:
        random_sampler(args, hdf)
