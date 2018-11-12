import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
# import IPython
# from collections import defaultdict
import pickle as pkl
import multiprocessing as mp
import argparse
import os
import collections


def random_crop(args, pid, img, mask=[]):
    """
    Create S random selection of a page with a fixed box size

    Parameters
    ----------
    args: argparser object specified in terminal command
    pid: pageid for saving out img (if specified in args)
    img: imread object of img (df["path"])
    mask: array with imread object of masks (df["mask"]) in bitwise_or

    Returns
    ----------
    pid: page id given in function arguments

    and if specified, saved cropped version of img
    """

    # creating empty lists to store pixel maps
    imgs = []
    labs = []

    it = 0
    h, w = img.shape[:2]

    # no handwritten elements on entire page
    if len(mask) < 1:
        print("skip")

    # handwritten elements contained on page
    else:
        # find pixels where HW mask is
        y, x = np.where(mask == 255)

        indices = np.linspace(0, args.box - args.hwbox,
                              (args.box - args.hwbox) / args.hwbox + 1).astype(int)

        sample = 0
        max_samples = int(150. * (len(x) * 1. / (h * w)))
        while sample < max_samples:

            # 1. select box
            # - sub-section of page with border size of box -
            rand = np.random.randint(0, high=len(x) - 1)
            rand_h, rand_w = y[rand], x[rand]

            # - from random pixel, create box with 1/4 directions
            rand_hs = (rand_h, rand_h + np.random.choice([-1, 1]) * args.box)
            rand_ws = (rand_w, rand_w + np.random.choice([-1, 1]) * args.box)
            area = args.box ** 2
            # make sure bounding box is within page
            if min(rand_hs) >= 0 and max(rand_hs) <= h and \
                    min(rand_ws) >= 0 and max(rand_ws) <= w:

                # - crop mask -
                crop_mask = mask[min(rand_hs):max(rand_hs),
                                 min(rand_ws):max(rand_ws)]

                sub_area = np.sum(crop_mask.flatten() == 255)

                if (sub_area * 1. / area) > 0.12:

                    crop_img = img[min(rand_hs):max(rand_hs),
                                   min(rand_ws):max(rand_ws)]
                    imgs.append(crop_img)

                    labels = []
                    for ys in indices:
                        for xs in indices:
                            sub_mask = crop_mask[ys:ys + args.hwbox,
                                                 xs:xs + args.hwbox]
                            sub = np.sum(sub_mask.flatten() == 255)

                            if sub * 1. / (args.hwbox**2) > 0.12:
                                labels.append(1)
                            else:
                                labels.append(0)
                    labs.append(labels)

                    sample += 1
    return imgs, labs


def mp_sampler(zipped):
    """
    Prepare for S samplings by loading img/masks & rescaling to pass onto
    random_crops()

    Parameters
    ----------
    zipped: zip object of grouped df and argparser object

    """

    # parsing zipped input
    grouped, args = zipped
    name, group = grouped

    # if more than one mask, path will be duplicated
    path = group["path"].unique()[0]

    # variable for if saving out random cropped images
    base = (os.path.normpath(path)).split(os.sep)[2]
    page_base = os.path.splitext(os.path.basename(path))[0]
    pageid = "%s_%s" % (base, page_base)

    # 0 import in image and masks
    img = cv2.imread(path)
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

    # TO DO| if save samp, create all sub-directories here so just called once
    hasHW = bool(group.hasHW.max())
    # if not hasHW:
    # # 1.a no masks are present; hasHW = 0
    # if "" in group["mask"].unique():
    #     # tup =
    #     random_crops(args, pageid, img, masks="")
    #     tup = "no hw"
    #     # 1.b has mask(s)

    # if not hasHW:
    #     print(name)
    if hasHW:
        or_mask = []
        # 2.a need to load each mask for cropping classification
        for index, el in group.iterrows():
            if el["hwType"] == "mach_sig":
                continue

            # otherwise, handwritten element
            mask = cv2.imread(el["mask"], 0)
            if len(or_mask) < 1:
                or_mask = mask
            else:
                # combine mark and text masks
                or_mask = cv2.bitwise_or(or_mask, mask)

        # scale mask to be same size of image
        or_mask = cv2.resize(or_mask, (0, 0), fx=scale_me, fy=scale_me)
        tup = random_crop(args, pageid, img, np.array(or_mask))

    return tup


def main(args):
    """
    Execute the multiprocessing of mp_sampler() and equalizer()

    Parameters
    ----------
    args: argparser object specified in terminal command

    """

    # reading in HDF of all labeled elements
    hdf_path = args.input[0]
    data = pd.read_hdf(hdf_path)

    # creating files to save out random selections

    # reducing data to first element to only load img/mask once
    sel = data.groupby(["pageid", "hwType", "path"], as_index=False).first()
    # then grouping by unique page identifiers
    grouped = sel.groupby(["pageid", "path"], as_index=False)
    hasHW = sel[sel["hasHW"] == 1]
    hasHW_group = hasHW.groupby(["pageid", "path"], as_index=False)

    # creating iterable version of args
    hasHWitems = [args] * len(hasHW_group)

    samples = []
    samp_labs = []
    samps = 0
    # 1. sample/ select all HW elements
    for s in zip(hasHW_group, hasHWitems):
        imgs, labs = mp_sampler(s)
        samps += len(imgs)
        if len(imgs) > 0:
            samples.extend(imgs)
            samp_labs.extend(labs)
    f = open("pixel_maps_%i.pkl" % samps, "wb")
    pkl.dump(samples, f)
    pkl.dump(samp_labs, f)
    f.close()
    # print(samps, len(samples))

    # for name, group in grouped:
    # print(name)

    # save one local node for sanity
    # pool = mp.Pool(mp.cpu_count() - 1)
    # num_pages = 0
    # num_pagesHW = 0
    # for count, result in enumerate(pool.imap(mp_sampler, zip(grouped, items))):
    #     print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample pages for CNN use.')

    parser.add_argument('samples', metavar='S', type=int, nargs=1,
                        help='number of desired samples per page')
    parser.add_argument('input', metavar='filename', type=str, nargs=1,
                        help='name of full HDF (not hasHW)')

    parser.add_argument('--box', dest='box', type=int,
                        default=150, help="int n (150) for creating n x n pixel \
                        box")
    parser.add_argument('--hwbox', dest='hwbox', type=int,
                        default=15, help="int n (15) for creating n x n pixel \
                        box for labeling if HW or not; must be int factor of box")
    parser.add_argument('--side', dest='side', type=int, default=20,
                        help='int s (20) for creating nested (n-s)x(n-s) box')
    parser.add_argument('--keep_machSig', dest='machSig', type=bool,
                        default=False, help="boolean to keep sampled image with machine \
                        signature or not (False)")
    parser.add_argument('--save_Samples', dest='save_samp', type=bool,
                        default=False, help='bool to save sampled images (png) \
                        or not (False)')
    parser.add_argument('--save_SampDir', dest='save_dir', type=str,
                        default="../data/samples/", help="directory to save random, \
                        cropped images")
    # padder(parser.parse_args, "equalSamp_26-10_s250_b150_5959.pkl")
    main(parser.parse_args())
