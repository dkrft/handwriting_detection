#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
This module provides the default values and functions that are used in
several scripts throughout the data module:
 * create_dataset
 * sampler/*
 * has_handwriting

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

This is the args that is referred to in the aforementioned scripts. For their
use, please refer to their usage examples.

"""


import argparse
import multiprocessing as mp
import numpy as np
import os


# --- convenience functions ---

# used to get basename for filepath
base = lambda path: os.path.splitext(os.path.basename(path))[0]

# filepath/name  for random_sampler() files; reused by other functions
rand_base = lambda args, base: "%s/randomSamp/rand_%s_samp%s_box%s_side%s" % \
    (args.saveDir, base, args.samples[0], args.box,
        args.side if not args.noSide else 0)

# filepath/name for data_mixer() files; reused by other functions
mix_base = lambda args, base, limit, noHW_limit: \
    "%s/mixSamp/mix_%s_HW%s_noHW%s_box%s_side%s.pkl" % \
    (args.saveDir, base, limit, noHW_limit, args.box,
        args.side if not args.noSide else 0)

# filepath/name for trainTest_set files; reused by other functions
train_base = lambda args, base, tot: \
    "%s/trainTest_%s_tot%s_box%s_side%s.pkl" % \
    (args.saveDir, base, tot, args.box, args.side if not args.noSide else 0)

# used to create grid index (has_handwriting); may be needed for CNN too
index = lambda box, gridBox: np.linspace(0, box - gridBox,
                                         (box - gridBox) / gridBox + 1).astype(int)


def get_default_parser():
    """
    Obtain default parser for script

    Returns
    -------
    argparse.ArgumentParser
        argument parser object

    """

    # create parser object; dictionary with variables used in script
    parser = argparse.ArgumentParser(description='Settings to sample pages for CNN use.',
                                     prog='PROG',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )

    # must give number of samples per page and HDF files with pages tabulated
    parser.add_argument('samples', type=int, nargs=1,
                        help='number of desired samples per page')
    # allows multiple inputs
    parser.add_argument('inputs', type=str, nargs="+",
                        help='list of full HDFs (not hasHW)')

    # --- general settings ---
    general = parser.add_argument_group("general settings")
    general.add_argument('--box', dest='box', type=int,
                         default=150, help="int n for creating n x n pixel \
                        sampling box")
    # Should each sample be evaluated for handwriting fully or in a grid of
    # sub-boxes? Default is consider full sample.
    general.add_argument('--grid', dest='grid', default=False,
                         action='store_true', help="mark sub-samples of box as \
                        having handwritten elements or not")
    general.add_argument("--mixFactor", dest="mixFactor", type=float,
                         default=1., help="float multiplier used to determine \
                        how many noHW elements to select in data_mixer() given \
                        the limit of the number of HW elements; applied at\
                        sample-level (not grid)")
    general.add_argument("--nproc", dest="nproc", type=int,
                         default=mp.cpu_count() - 1, help="int for number of \
                        processes to simultaneously run (machine - 1)")
    # assumes that soft-linked data directory is +1 outside of full git dir
    # so +2 ../../ outside of hwdetect/data
    general.add_argument("--imgDir", dest='imgDir', type=str,
                         default="../../data/original_data/", help="path to \
                        directory of original images and masks")
    # where and how to save samples & labels
    general.add_argument("--saveDir", dest='saveDir', type=str,
                         default="../../data/training_data/",
                         help="path to directory to save sampling files")
    general.add_argument('--debug', dest="debug", default=False,
                         action='store_true', help="use matplotlib to display \
                        potential HW elements (parallel processing disabled)")

    # settings if grid = False
    box = parser.add_argument_group("if label per box")

    # how should HW elements be detected?
    # should it be if the HW mask is present in centered box inside the sample?
    # default: yes; else based on the number of non-white pixels above a
    # threshhold
    box.add_argument('--noSide', dest='noSide', default=False,
                     action='store_true', help='do not evaluate sub-, \
                        centered box within box as having handwriting or not; \
                        paired with --side')
    box.add_argument('--side', dest='side', type=int, default=50,
                     help='int s for creating centered (n-2s)x(n-2s) box ')

    grid = parser.add_argument_group("if evaluate grid of box")
    # how should HW elements be detected?
    #  * in centered sub-box inside grid cell? (default=True)
    #  * number of non-white pixels above a threshhold
    grid.add_argument('--gridBox', dest='gridBox', type=int,
                      default=15, help="int n for n x n pixel grid cell in sample \
                        for labeling if HW or not; must be int factor of box size")
    grid.add_argument('--noGridSide', dest='noGridSide', default=False,
                      action='store_true', help='do not evaluate sub-, \
                        centered box within each grid-box as having handwriting \
                        or not')
    grid.add_argument('--gridSide', dest='gridSide', type=int, default=5,
                      help='int s for creating nested (n-2s)x(n-2s) box ')

    # debug only refers to images that had masks with some form of handwriting
    debug = parser.add_argument_group("settings for debug")
    debug.add_argument('--showAll', dest='showAll', default=False,
                       action='store_true', help="show all image plots; even \
                       when  selected image/mask contains no handwriting")
    debug.add_argument('--showCells', dest='showCells', default=False,
                       action='store_true', help="show individual cells")
    debug.add_argument('--saveData', dest='saveData', default=False,
                       action='store_true', help="while running debug save \
                       data")

    # # hdf of saved data for stats
    # # fast save with just sampling handwriting

    # # # selection criteria for samples
    # # parser.add_argument('--keep_machSig', dest='machSig', type=bool,
    # #                     default=False, help="boolean to keep sampled image \
    # #                     with machine signature or not (False)")
    return parser
