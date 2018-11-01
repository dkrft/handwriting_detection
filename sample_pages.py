from skimage import draw, io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import IPython
from collections import defaultdict

tuner = defaultdict(list)


def random_crop(dims, img, mask, samples=30, box=100, side=20):
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
        # io.imshow(crop_img)
        # plt.show()

        # TO DO apply cropped mask and variance threshold (NEED TO STUDY)
        # or keep within center of box with side modifications
        # TO TUNE FURTHER; about 1 out of every 500 false positive
        # typically pure white or close to it
        # threhold value of 60 helps but the passing ones
        # range from 300-700 positive pixels which are in line with positives

        crop_mask = np.array(mask[rand_h:rand_h + box,
                                  rand_w:rand_w + box])

        # has indent
        sub_mask = np.array(mask[rand_h + side:rand_h + box - side,
                                 rand_w + side:rand_w + box - side])

        # TO DO decide on threshold value or use any to expedite search
        whites = (sub_mask == [0, 0, 0, 255]).all(-1)

        if sum(~whites.flatten()) > 0:

            # creates mask from mask to identify black pixels
            img_mask = (crop_mask == [0, 0, 0, 255]).all(-1)
            # filter cropped image to black-out all but the selected pixels
            masked_img = np.where(~img_mask[..., None], crop_img, 0)

            # look at image and filtered image
            io.imshow(crop_img)
            plt.show()

            io.imshow(masked_img)
            plt.show()

            tuner["sum_whites_sub"].append(sum(~whites.flatten()))
            tuner["sum_whites"].append(sum(~img_mask.flatten()))
            # tuner["true"].append(input('Good (y) / bad (n):'))

    print(tuner)

data = pd.read_hdf("labels/26-10.hdf")

# reducing data to first element to only load img/mask once
sel = data.groupby(["pageid", "hwType"], as_index=False).first()

dims = defaultdict(list)
for index, row in sel.iterrows():

    # 0 import in image and mask
    img = io.imread(row["path"])
    h, w = img.shape[:2]
    dims["height"].append(h)
    dims["width"].append(w)

    # only need to check mask if hasHW
    if row.hwType != "":
        mask = io.imread(row["mask"])
        m_h, m_w = mask.shape[:2]
        dims["m_h"].append(m_h)
        dims["m_w"].append(m_w)

        # 1: select random box(es) for CNN
        random_crop((h, w), img, mask)

    break

# TO DO | 2D histogram of page dimensions
