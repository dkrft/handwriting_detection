#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
This script samples the pages listed in a pandas dataframe to generate
pickled classes (TrainingData) for use in training a convolutional neural
network (CNN)

For successful usage, the dataframe must contain the following columns:
    hwType, hasHW, pageid, path, and mask
with values as described in about_hdf.md. [Files produced by handle_Labelbox.py
and read_PRImA.py already meet these requirements.]

For the optimization of the CNN, this script has been oufitted with a modular
design and is easily adapted with several key arguments contained within a
default dictionary (get_default_parser()).

The main functions used within this script are:
 * random_sampler() --- samples each page and separately saves images that
    contain or do not contain handwritten elements in pickled dictionaries.
    The dictionaries contain lists of the pixel maps and labels, as well
    as other identifying information.

 * data_mixer() --- generates a specific mix of samples that contain
    handwriting or not. The files generated from the random_sampler() are
    expected inputs.

 * trainTest_set() --- generates a shuffled training and test set for use in
    the CNN

Intended for usage in the terminal like:

python create_dataset.py 250 ../../data/labeled_databases/26-10.hdf

OR

python sample_pages.py 250 ./../data/labeled_databases/26-10.hdf --box 100 --side 20

To use in script directly:
>>> import hwdetect.data.create_dataset
>>> from hwdetect.data.defaults import get_default_parser()

Need to obtain default parameters for generating data
>>> parser = get_default_parser()

To view defined keys in argparser:
>>> parser.print_help()

# To modify optional parser objects
>>> parser.grid = True

To pass the required information (number of samples per page and path to HDF)
>>> args = parser.parse_args(['1', "../../data/labeled_databases/26-10.hdf"])

To run function to randomly select, mix, and separate training and testing data
>>> create_dataset.main(args)

"""

import os
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split

from hwdetect.data.training_data import TrainingData
from defaults import base, get_default_parser, mix_base, rand_base, train_base
from sampler import random_sampler


def data_mixer(args, hdf_path):
    """
    Create mixed sample of samples with and without handwritten elements;
    number of handwritten samples is the limiting factor multiplied by the
    args.mixFactor to determine the number of samples without handwriting

    Parameters
    ----------
    args : argparse.ArgumentParser
        argparser object specified in get_default_parser (& terminal)
    hdf_path : str
        string with full or relative path to an HDF pandas dataframe

    Outputs
    ----------
    mix*.pkl
        pickled file with specified selection from
        random*.pkl of cropped images with or without handwritten text;
        need to randomly select from/shuffle as ordered data

    """

    filebase = rand_base(args, base(hdf_path))

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
    mem_safety_factor = max(1, (num_noHW * args.samples[0] * args.box**2) //
                            (5000 * 150**2))

    group_size = num_noHW // mem_safety_factor
    sel_size = noHW_limit // mem_safety_factor

    g = open("%s_noHW.pkl" % (filebase), "rb")
    jt = 1
    for page in range(1, num_noHW):
        if jt < group_size and jt != num_noHW:
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

    name = mix_base(args, base(hdf_path), limit, noHW_limit)
    if len(np_pixels) == len(np_labels):
        h = open(name, 'wb')
        pkl.dump({"imgs": np_pixels, "labels": np_labels}, h)
        h.close()

        print("data_mixer succeeded! \nFile %s" % name)

    else:
        print("ERROR in data_mixer()")

    f.close()
    g.close()
    return name, (limit + noHW_limit)


def trainTest_set(filelist, savepath):
    """
    Prepare training and test data from one or more pickled files that have
    been generated with dictionaries containing the keys "imgs" and "labels"
    (sampler/random_sampler or data_mixer)

    Parameters
    ----------
    filelist : list
        list of full or relative paths to mixSamp or randomSamp file(s)
    savepath : str
        path and name of pickled file to save with training and test data

    Outputs
    ----------
    trainTest*.pkl
        pickled dictionary with "x_train", "x_test", "y_train", "y_test";
        randomized and shuffled data from filelist's files

    """
    pixels = []
    labels = []
    for file in filelist:
        f = open(file, "rb")
        data = pkl.load(f)
        pixels.extend(data["imgs"])
        labels.extend(data["labels"])
        f.close()

    x_train, x_test, y_train, y_test = train_test_split(pixels, labels,
                                                        test_size=0.30,
                                                        random_state=42)

    f = open(savepath, "wb")
    data_class = TrainingData(x_train, y_train, x_test, y_test)
    pkl.dump(data_class, f)
    f.close()
    print("Training and testing data saved to: %s" % savepath)


def main(args):
    """
    Execute the functions sequentially in order to obtain a
    training and test data set for a convolutional neural network:
     1. randomly sample each page of a given file with:
            random_sampler()
     2. from files of random samples create balanced mixture
        of samples with and without handwriting:
            data_mixer()
     3. from prepared mixtures prepare training and test data for CNN:
            trainTest_sets()

    Parameters
    ----------
    args: argparser object specified in terminal command

    Outputs
    ----------
    rand*.txt: text file
        with # of pickled files
    rand*.pkl: pickled file
        dictionaries for each page with all the random samples; for ea. page,
        save list with all pixel maps & list with all labels
    mix*.pkl: pickled file
        dictionary with specified mixed selection from random*.pkl
        of cropped images with or without handwritten text; need to randomly
        select from/shuffle as ordered data
    trainTest*.pkl
        pickled dictionary with "x_train", "x_test", "y_train", "y_test";
        randomized and shuffled data from filelist's files

    """

    mixed_files = []
    bases = []
    vals = []
    for file in args.inputs:

        # 1. if not existing, randomly select X n x n samples from each page
        #    listed in the HDF dataframe
        random = rand_base(args, base(file)) + ".txt"
        if not os.path.isfile(random):
            random_sampler(args, file)
        else:
            print("\nRandom selection of %s already exists" % file)

        # 2. mix random samples with and without handwriting on a file basis
        #   to obtain the desired proportion of samples
        print("\nCreating mixed sample")
        mix_file, val = data_mixer(args, file)
        mixed_files.append(mix_file)
        bases.append(base(file))
        vals.append(val)

    # 3. create training and test sets
    print("\nCreating training and test set for CNN")
    train_name = train_base(args, "_".join(bases), sum(vals))
    trainTest_set(mixed_files, train_name)


if __name__ == '__main__':
    parser = get_default_parser()
    # parser.print_help()
    main(parser.parse_args())
