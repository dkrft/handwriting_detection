import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
# import IPython
import pickle as pkl
import multiprocessing as mp
import argparse
import os
from collections import Counter, defaultdict
import operator
import scipy.ndimage as ndimage


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
    neighs = []

    # creating dictionary to store values for analytics
    holder = defaultdict(list)

    it = 0
    h, w = img.shape[:2]

    footprint = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])

    def WeightFunc(a):
        a = a.reshape((3, 3))
        return np.average(a, weights=np.array([[0.5, 0.5, 0.5], [0.5, 2, 0.5],
                                               [0.5, 0.5, 0.5]]))

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

                    labels = []
                    for ys in indices:
                        row = []
                        for xs in indices:
                            sub_mask = crop_mask[ys:ys + args.hwbox,
                                                 xs:xs + args.hwbox]
                            sub = np.sum(sub_mask.flatten() == 255)

                            sub_img = crop_img[ys:ys + args.hwbox,
                                               xs:xs + args.hwbox]
                            gray_img = cv2.bitwise_not(
                                cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY))
                            mask_img = cv2.bitwise_and(gray_img, gray_img,
                                                       mask=sub_mask)

                            gvals_dic = Counter(gray_img.flatten())
                            mvals_dic = Counter(mask_img.flatten())
                            mvals = [item for key,
                                     item in mvals_dic.items() if key > 15]

                            holder["pageid"].append(pid)
                            holder["sampID"].append(sample)
                            holder["x_min"].append(min(rand_ws))
                            holder["x_max"].append(max(rand_ws))
                            holder["y_min"].append(min(rand_hs))
                            holder["y_max"].append(max(rand_hs))
                            holder["nonzeros"].append(sum(mvals))
                            holder["sub_y"].append(ys)
                            holder["sub_x"].append(xs)
                            # holder["mvals"].append(mvals_dic)

                            if sub > 0 and sum(mvals) > 10:
                                if args.debug:
                                    print("gray_img:")
                                    print(sorted(gvals_dic.items(),
                                                 key=operator.itemgetter(0)))

                                    print("mask_img:")
                                    print(sorted(mvals_dic.items(),
                                                 key=operator.itemgetter(0)))

                                    print("Non-zeros: %i" % sum(mvals))

                                    fig, axes = plt.subplots(nrows=2, ncols=2, )

                                    plt.setp(axes[0, 0].get_xticklabels(),
                                             fontsize=10)
                                    axes[0, 0].imshow(sub_img, vmin=0, vmax=255)
                                    axes[0, 0].set_title("Image")

                                    axes[0, 1].imshow(sub_mask, vmin=0,
                                                      vmax=255)
                                    axes[0, 1].set_title("Mask")

                                    axes[1, 0].imshow(gray_img, vmin=0,
                                                      vmax=255)
                                    axes[1, 0].set_title("Grayscale image")

                                    axes[1, 1].imshow(mask_img, vmin=0,
                                                      vmax=255)
                                    axes[1, 1].set_title("Masked grayscale \
                                        image")
                                    plt.show()

                                row.append(1)
                                holder["hasHW"].append(1)
                            else:
                                row.append(0)
                                holder["hasHW"].append(0)
                        labels.append(row)

                    strict_labels = np.array(labels, dtype=float)
                    neigh_labels = ndimage.generic_filter(strict_labels,
                                                          WeightFunc,
                                                          footprint=footprint,
                                                          mode='nearest')

                    imgs.append(crop_img)
                    labs.append(strict_labels.flatten())
                    neighs.append(neigh_labels.flatten())
                    sample += 1

    return imgs, labs, neighs, holder


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
    if not hasHW:
        tup = random_crop(args, pageid, img, [])
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
    items = [args] * len(grouped)

    samples = []
    samp_labs = []
    samp_neighs = []
    samps = 0

    f = open("pixel_maps_%i.pkl" % samps, "wb")

    # 1. sample/ select all HW elements
    for it, s in enumerate(zip(grouped, items)):
        imgs, str_labs, neigh_labs, stats = mp_sampler(s)

        # if it == 0:
        #     data = pd.DataFrame(stats)
        # else:
        #     df = pd.DataFrame(stats)
        #     data = data.append(df)

        samps += len(imgs)
        if len(imgs) > 0:
            samples.extend(imgs)
            samp_labs.extend(str_labs)
            samp_neighs.extend(neigh_labs)

    # 2. sample/select without caring about HW elements
    # print(samps)
    # groupitems = [args] * len(grouped)
    # for g in zip(grouped, groupitems):
    #     if

    data.to_hdf("statistics.hdf", key="data")

    pkl.dump(samples, f)
    pkl.dump(samp_labs, f)
    pkl.dump(neigh_labs, f)
    f.close()

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
    parser.add_argument('--debug', dest='debug', type=bool,
                        default=False, help="show images on screen to debug data\
                       production")
    # padder(parser.parse_args, "equalSamp_26-10_s250_b150_5959.pkl")
    main(parser.parse_args())
