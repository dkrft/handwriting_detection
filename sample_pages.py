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


def random_crops(args, pid, img, masks=""):
    """
    Create S random selections of a page with a fixed box size

    Parameters
    ----------
    args: argparser object specified in terminal command
    pid: pageid for saving out img (if specified in args)
    img: imread object of img (df["path"])
    masks: array with imread object of masks (df["mask"])

    Returns
    ----------
    pid: page id given in function arguments
    pmap_noHW: array appended to with pixel maps without a handwritten element
    pmap_HW: array appended to with pixel maps with a handwritten element

    and if specified, saved cropped version of img
    """

    # creating empty lists to store pixel maps
    pmap_noHW = []
    pmap_HW = []

    it = 0
    h, w = img.shape[:2]
    # sample page; use while loop as some types of elements may be rejected
    while it < args.samples[0]:
        # - bottom left pixel of box -
        rand_h = np.random.randint(0, h - args.box)
        rand_w = np.random.randint(0, w - args.box)

        crop_img = img[rand_h:rand_h + args.box, rand_w:rand_w + args.box]

        # has no mask or hw element
        if isinstance(masks, str):
            if args.save_samp:
                cv2.imwrite(args.save_dir + "no_mask/%s_%i.png" %
                            (pid, it), crop_img)
            pmap_noHW.append(crop_img)
            it += 1

        # page has mask(s)
        else:
            # TO DO| consider changes to cropped mask (e.g. variance);
            # code does not distinguish between mark/text/etc. mixed cases
            # based on first mask found in area meeting criteria
            hasMask = False
            for hwType, in_mask in masks.items():
                # inset of mask box to ensure HW element (label not on edge)
                sub_mask = np.array(in_mask[rand_h + args.side:rand_h + args.box - args.side,
                                            rand_w + args.side:rand_w + args.box - args.side])

                # checks to see if mask present in cropped area:
                # use no. white (black in cv2) pixels > 0 to distinguish
                if sum(sub_mask.flatten() == 255) > 0:
                    # to check on masking (but not in parallel processes)
                    # crop_mask = np.array(in_mask[rand_h:rand_h + args.box,
                    #                              rand_w:rand_w + args.box])
                    # mask_img = cv2.bitwise_and(crop_img, crop_img,
                    #                            mask=crop_mask)
                    # plt.imshow(mask_img)
                    # plt.show()

                    # sometimes we don't want to keep machine signatures
                    # TO DO| If keep machSig, need to check that HW elems never overlap
                    # or need to more robustly code
                    if hwType == "mach_sig":
                        if args.machSig:
                            if args.save_samp:
                                cv2.imwrite(args.save_dir + "mask/%s/%s_%i.png" %
                                            (hwType, pid, it), crop_img)
                            hasMask = True
                            pmap_noHW.append(crop_img)
                            it += 1
                            break

                        # don't want to save these; don't count this sample
                        else:
                            # continue as may have other masks it works with
                            continue
                    # assumed either text or mark; save out & new sample
                    else:
                        if args.save_samp:
                            save_path = args.save_dir + "mask/%s/%s_%i.png" % \
                                (hwType, pid, it)
                            cv2.imwrite(save_path, crop_img)

                        pmap_HW.append(crop_img)
                        hasMask = True
                        it += 1
                        # found a matching mask; no need to consider others
                        break

            # had mask but not included in crop
            if not hasMask:
                if args.save_samp:
                    cv2.imwrite(args.save_dir + "mask/not_in_samp/%s_%i.png" %
                                (pid, it), crop_img)

                pmap_noHW.append(crop_img)
                it += 1

    return pid, pmap_noHW, pmap_HW


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
    _, group = grouped

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
    h, w = img.shape[:2]

    # TO DO| if save samp, create all sub-directories here so just called once

    # 1.a no masks are present; hasHW = 0
    if "" in group["mask"].unique():
        tup = random_crops(args, pageid, img, masks="")
    # 1.b has mask(s)
    else:
        masks = {}
        # 2.a need to load each mask for cropping classification
        for index, el in group.iterrows():
            # comes in inverted from saved png
            mask = cv2.imread(el["mask"], 0)
            mask = cv2.resize(mask, (0, 0), fx=scale_me, fy=scale_me)
            masks[el["hwType"]] = mask
        tup = random_crops(args, pageid, img, masks)

    return tup


def equalizer(date, HW, num_HW, noHW, num_noHW):
    """
    Equally select HW and noHW elements using num_HW
    as sample size for each population

    Parameters
    ----------
    HW: name of pickled file with HW elements
    num_HW: # of added HW elements
    noHW: name of pickled file with noHW elements
    num_noHW: # of added noHW elements

    Returns
    ----------
    equalSamp*.pkl: pickled file with equal selection from random*.pkl
                    of cropped images with or without handwritten text;
                    need to randomly select from/shuffle as ordered data

    """
    pixels = []
    page = []
    f = open(HW, "rb")
    for it in range(num_HW):
        pixels.extend(pkl.load(f))
        page.extend(pkl.load(f))

    # how many samples of HW and noHW we want
    limit = len(pixels)
    labels = [1] * len(pixels)
    pixels = np.array(pixels)

    # performing noHW selection in 5 groups with remainders ignored
    jt = 1
    objects = []
    g = open(noHW, "rb")
    group_size = num_noHW // 4
    sel_size = limit // 4

    for page in range(num_noHW):
        if jt < group_size and jt != num_noHW:
            objects.extend(pkl.load(g))
            _ = pkl.load(g)
            jt += 1

        if jt == group_size:
            print("Processing %s objects; selecting %s" %
                  (len(objects), sel_size))
            np_objs = np.array(objects).copy()
            rand_sel = np.random.randint(0, high=len(objects) - 1,
                                         size=sel_size)
            sel_noHW = np_objs[rand_sel]

            # appending to HW elems
            pixels = np.append(pixels, sel_noHW, axis=0)
            labels = np.append(labels, [0] * len(sel_noHW))

            # starting over
            jt = 1
            objects = []

    if len(pixels) == len(labels) and np.abs(len(pixels) - 2 * limit) < 4:
        print("Equalizer succeeded!")
        h = open("equalSamp_%s_%s.pkl" % (date, limit), 'wb')
        pkl.dump(pixels, h)
        pkl.dump(labels, h)
        h.close()
    else:
        print("ERROR in equalizer()")

    f.close()
    g.close()


def padder(args, equalfilepath):
    """
    Add padding of various widths to equalized sample

    Parameters
    ----------
    args: argparser object specified in terminal command
    equalfilepath: path to equalized sample

    Returns
    ----------
    paddedSamp*_pxxx.pkl: pickled file with equal selection from equalSamp*.pkl
                    of cropped images with specified padding,
                    with or without handwritten text;
                    need to randomly select from/shuffle as ordered data
    """

    start, mid1, mid2, end = equalfilepath.split("_")[1:5]
    f = open(equalfilepath, 'rb')
    pixels = pkl.load(f)
    labels = pkl.load(f)
    f.close()

    WHITE = [255, 255, 255]
    padding = [10, 30, 60, 100]
    for p in padding:
        new_pixels = [0] * len(labels)
        save_to = "%s_%s_%s_p%s_%s" % (start, mid1, mid2, p, end)

        for it, img in enumerate(pixels):
            new_pixels[it] = cv2.copyMakeBorder(img, p, p, p, p,
                                                cv2.BORDER_CONSTANT,
                                                value=WHITE)
        g = open(save_to, "wb")
        pkl.dump(new_pixels, g)
        pkl.dump(labels, g)
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

    # creating files to save out random selections
    base = os.path.splitext(os.path.basename(hdf_path))[0]
    HW_file = "random_%s_%sea_HW.pkl" % (base, args.samples[0])
    noHW_file = "random_%s_%sea_noHW.pkl" % (base, args.samples[0])

    f_HW = open(HW_file, "wb")
    g_noHW = open(noHW_file, "wb")

    # reducing data to first element to only load img/mask once
    sel = data.groupby(["pageid", "hwType", "path"], as_index=False).first()
    # then grouping by unique page identifiers
    grouped = sel.groupby(["pageid", "path"], as_index=False)
    # creating iterable version of args
    items = [args] * len(grouped)

    # save one local node for sanity
    pool = mp.Pool(mp.cpu_count() - 1)
    num_pages = 0
    num_pagesHW = 0
    for count, result in enumerate(pool.imap(mp_sampler, zip(grouped, items))):
        pid, pmap_noHW, pmap_HW = result

        if count % 10 == 0:
            print("%s pages processed" % count)

        if len(pmap_noHW) > 0:
            pkl.dump(pmap_noHW, g_noHW)
            pkl.dump(pid, g_noHW)
            num_pages += 1

        if len(pmap_HW) > 0:
            pkl.dump(pmap_HW, f_HW)
            pkl.dump(pid, f_HW)
            num_pagesHW += 1

    f_HW.close()
    g_noHW.close()

    # print(num_pagesHW, num_pages)
    equalizer(base, HW_file, num_pagesHW, noHW_file, num_pages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample pages for CNN use.')

    parser.add_argument('samples', metavar='S', type=int, nargs=1,
                        help='number of desired samples per page')
    parser.add_argument('input', metavar='filename', type=str, nargs=1,
                        help='name of full HDF (not hasHW)')

    parser.add_argument('--box', dest='box', type=int,
                        default=150, help="int n (150) for creating n x n pixel \
                        box")
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
    padder(parser.parse_args, "equalSamp_26-10_s250_b150_5959.pkl")
    # main(parser.parse_args())
