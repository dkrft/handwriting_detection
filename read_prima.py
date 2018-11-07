"""Convenience functs for creating HDF and masks for PRiMA NHM data"""

from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from xml.etree import ElementTree

source_dir = "../data/PRiMA/"


def make_maskHDF(filebase):
    """Create mask and HDF element for each PRiMA file

    Parameters
    ----------
    filebase: base name for file associated with img, xml
              and mask to be created

    Returns
    ----------
    maskpath: path to mask for storage in the hdf
    mask png saved to ../data/PRimA/text_mask

    """
    xmlpath = source_dir + "labels/%s.%s" % (filebase, "xml")

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
                    geo[points].append(child.attrib)

        # creating mask
        img = np.zeros((height, width, 3), np.uint8)
        # checked: no 0 lengths; each page has at least 1 handwritten elem
        for group, pts in holder.items():
            # checked: no <=2 lengths of pts; all polygon approved
            # formatting to acceptable input
            form_pts = np.array(pts, np.int32)
            form_pts = form_pts.reshape((-1, 1, 2))
            img = cv2.fillPoly(img, [form_pts], color=(255, 255, 255))

        maskpath = source_dir + "text_mask/%s.%s" % (filebase, "png")
        cv2.imwrite(maskpath, img)
        return maskpath

prima = defaultdict(list)
for file in os.listdir(source_dir + "img"):
    base = os.path.splitext(file)[0]
    maskpath = make_maskHDF(base)
    prima["mask"]
    break
    # make_maskHDF()
    # source_dir + "img/" + file
