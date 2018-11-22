#!/usr/bin/env python

from distutils.core import setup

setup(
    name='hwdetect',
    version='0.1',
    description='detects handwriting',
    install_requires=[
        "pyparsing>=2.1.4",
        "pandas>=0.19.1",
        "scipy>=0.18.1",
        "nltk>=3.2.1",
        "gensim>=0.13.0",
        "tqdm>=4.10.0",
        "numpy>=1.11.2",
        "cytoolz>=0.8.2",
    ]
)