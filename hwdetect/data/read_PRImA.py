"""Script for for processing the PRiMA NHM data

This script recursively goes through a directory containing the PRImA NHM
data from:

https://www.primaresearch.org/datasets/NHM

where the images (*.tif) are in a sub-directory called img and the xml files
(containing the coordinates demarking where handwriting is located is given in
a sub-directory) called labels. The script then generates a black-and-white
image, known as a mask, where white polygons are formed by the coordinates
given in the xml files and thus demark areas with handwritten elements.

In order to sample or select pages, a pandas dataframe is created and stored
in an HDF. The provided columns are described in the about_hdf.text file.

Usage
-------
Create HDF pandas dataframe of and masks for the PRImA data.
1. Set-up directory as described above.
2. Use commands given below with the path to the PRImA data and where you'd
   like the pandas HDF to be saved to.

>>> from hwdetect.data import read_PRImA
>>> read_PRImA.process_prima("../../data/original_data/PRImA",
                             "../../data/labeled_databases")
HDF saved to ../../data/labeled_databases/prima.hdf

"""

from collections import defaultdict
import os
from xml.etree import ElementTree

import cv2
import pandas as pd
import numpy as np

__author__ = "Ariel Bridgeman"
__version__ = "1.0"


def make_maskHDF(filebase, prima_dir):
    """Create and give path for mask generated a PRiMA file

    Parameters
    ----------
    filebase: str
        base name for a file associated with image and xml
        (and to-be-created mask)
    prima_dir: str
        path to directory containing PRImA images and XMLs

    Outputs
    ----------
    png
        mask png saved to prima_dir + "/text_mask/"

    Returns
    ----------
    maskpath: str
        path to mask for storage in the HDF
    geo: dict
        dictionary containing points bounding the handwritten elements
    """

    xmlpath = prima_dir + "/labels/%s.%s" % (filebase, "xml")

    if os.path.isfile(xmlpath):
        x = open(xmlpath)
        file = ""
        save = False
        for line in x.readlines():
            if "Page" in line:
                save = not save
                if "</Page>" in line:
                    line = line.split("</Page>")[0]
                    line += "</Page>"
                file += line
                continue
            if save:
                file += line
        x.close()

        xml = ElementTree.ElementTree(ElementTree.fromstring(file))

        # getting pixel dimensions for creating mask
        root = xml.getroot().attrib
        width = int(root["imageWidth"])
        height = int(root["imageHeight"])

        # looking for tags with handwritten-notation
        HW = False
        points = 0
        holder = defaultdict(list)
        geo = defaultdict(list)
        for coords in xml.findall("./GraphicRegion"):
            for child in coords.iter():
                if child.tag == "GraphicRegion":
                    if child.attrib["type"] == 'handwritten-annotation':
                        points += 1
                        HW = True
                        continue
                    else:
                        HW = False

                if child.tag == "Point" and HW:
                    holder[points].append([int(child.attrib["x"]),
                                           int(child.attrib["y"])])
                    geo[points].append({"x": int(child.attrib["x"]),
                                        "y": int(child.attrib["y"])})

        # creating mask
        img = np.zeros((height, width, 3), np.uint8)
        # checked: no 0 lengths; each page has at least 1 handwritten elem
        for group, pts in holder.items():
            # checked: no <=2 lengths of pts; all polygon approved
            # formatting to acceptable input
            form_pts = np.array(pts, np.int32)
            form_pts = form_pts.reshape((-1, 1, 2))
            img = cv2.fillPoly(img, [form_pts], color=(255, 255, 255))

        maskpath = prima_dir + "/text_mask/%s.%s" % (filebase, "png")
        cv2.imwrite(maskpath, img)
        return maskpath, geo


def process_prima(prima_dir, hdf_dir):
    """Process all PRImA files to create masks and generate a
    pandas dataframe for quick look-ups

    Parameters
    ----------
    prima_dir: str
        path to directory containing PRImA images and XMLs

    hdf_dir: str
        path to directory to save the pandas dataframe HDF in

    Outputs
    ----------
    png
        mask pngs saved to prima_dir + "/text_mask/"; generated
        from the coordinates given in the xmls
    hdf
        pandas dataframe HDF saved within hdf_dir


    """
    prima = defaultdict(list)
    img_dir = prima_dir + "/img"
    for file in os.listdir(img_dir):
        base = os.path.splitext(file)[0]
        maskpath, geo = make_maskHDF(base, prima_dir)
        for key, g in geo.items():
            # prima["geometry"].append(g)
            prima["hwType"].append("text")
            prima["hasHW"].append(1)
            prima["pageid"].append(file)
            prima["path"].append(prima_dir + "/img/%s" % (file))
            prima["mask"].append(maskpath)

    df = pd.DataFrame(prima)
    hdf_path = hdf_dir + "/prima.hdf"
    df.to_hdf(hdf_path, key="data")
    print("HDF saved to %s" % hdf_path)
