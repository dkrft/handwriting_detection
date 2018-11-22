#!/usr/bin/env python

from distutils.core import setup

setup(
    name='hwdetect',
    version='0.1',
    description='detects handwriting',
    install_requires=[
        "scipy>=0.18.1",
        "numpy>=1.11.2",
        "Django>=2.1.3",
        "opencv-python>=3.4.3.18",
        "tensorflow>=1.12.0",
    ]
)