from skimage import draw, io
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import IPython
from collections import defaultdict

tuner = defaultdict(list)
save_dir = "../data/samples/"


def random_crop(id, dims, img, mask="", samples=10, box=100, side=15):
    """
    Create x random selections of a page with a fixed box size

    Parameters
    ----------
    dims : dimensions of the image being considered

    img: imread object of img (df["path"])

    mask: imread object of mask (df["mask"])

    samples : number of times to sample from a page;
             should depend some on page size, no?

    box: integer pixels for height and width of box; assumed square

    side: integer pixel border subtracted from box to ensure mask element
          contained (not edge pixels); fully substracted from each side of dim

    Returns
    ----------
    saved cropped version of img

    """

    h, w = dims

    for it in range(samples):
        # - bottom left pixel of box -
        rand_h = np.random.randint(0, h - box)
        rand_w = np.random.randint(0, w - box)

        crop_img = img[rand_h:rand_h + box, rand_w:rand_w + box]

        if mask == "":
            cv2.imwrite(save_dir + "no_mask/%s_%i.png" % (pageid, it), crop_img)

        # TO DO apply cropped mask and variance threshold (NEED TO STUDY)
        # or keep within center of box with side modifications
        # TO TUNE FURTHER; about 1 out of every 500 false positive
        # typically pure white or close to it
        # threhold value of 60 helps but the passing ones
        # range from 300-700 positive pixels which are in line with positives
        else:

            crop_mask = np.array(mask[rand_h:rand_h + box,
                                      rand_w:rand_w + box])
            mask_img = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)

            whites = sum(mask_img.flatten() == 0)

            # has indent
            sub_mask = np.array(mask[rand_h + side:rand_h + box - side,
                                     rand_w + side:rand_w + box - side])

            # TO DO decide on threshold value or use any to expedite search
            sub_whites = sum(sub_mask.flatten() == 255)

            if sub_whites > 0:

                # look at image
                # plt.imshow(crop_img)
                # plt.show()

                # look at filtered image
                plt.imshow(mask_img)
                plt.show()

                tuner["sum_whites_sub"].append(sub_whites)
                tuner["sum_whites"].append(whites)

                # excludes mask
                mean, stdDev = cv2.meanStdDev(crop_img, mask=crop_mask)

                tuner["mean_r"].append(mean[0])
                tuner["mean_g"].append(mean[1])
                tuner["mean_b"].append(mean[2])

                tuner["std_r"].append(stdDev[0])
                tuner["std_g"].append(stdDev[1])
                tuner["std_b"].append(stdDev[2])

                # includes mask elements
                mean2, stdDev2 = cv2.meanStdDev(crop_img)

                tuner["mean_r_bkg"].append(mean2[0])
                tuner["mean_g_bkg"].append(mean2[1])
                tuner["mean_b_bkg"].append(mean2[2])

                tuner["std_r_bkg"].append(stdDev2[0])
                tuner["std_g_bkg"].append(stdDev2[1])
                tuner["std_b_bkg"].append(stdDev2[2])

                path = save_dir + "hw_mask/in_samp/%s_%i.png" % (pageid, it)
                tuner["path"].append(path)

                tuner["pen"].append(input('mark y / n?'))
                tuner["char"].append(input('char(s) y / n?'))
                tuner["edge"].append(input('on edge y / n?'))

                cv2.imwrite(path, crop_img)
            else:
                cv2.imwrite(save_dir + "hw_mask/not_in_samp/%s_%i.png" %
                            (pageid, it), crop_img)


data = pd.read_hdf("labels/26-10.hdf")

# reducing data to first element to only load img/mask once
sel = data.groupby(["pageid", "hwType"], as_index=False).first()

dims = defaultdict(list)
for index, row in sel.iterrows():
    pageid = (row["path"].split(".jpg")[0]).split("/")[2:5]
    pageid = "_".join(pageid)

    # 0 import in image and mask
    img = cv2.imread(row["path"])
    h, w = img.shape[:2]
    dims["height"].append(h)
    dims["width"].append(w)

    # only need to check mask if hasHW
    if row.hwType != "":

        # comes in inverted from saved png
        mask = cv2.imread(row["mask"], 0)
        m_h, m_w = mask.shape[:2]
        dims["m_h"].append(m_h)
        dims["m_w"].append(m_w)

        # 1: select random box(es) for CNN
        random_crop(pageid, (h, w), img, mask)
    else:
        random_crop(pageid, (h, w), img)

    if index % 5 == 0:
        samples = pd.DataFrame(tuner)
        samples.to_hdf("threshold.hdf")
        tuner = defaultdict(list)
        if index == 10:
            break

dimensions = pd.DataFrame(dims)
dimensions.to_hdf("page_dims.hdf")


# TO DO | 2D histogram of page dimensions
