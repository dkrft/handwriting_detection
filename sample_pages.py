from skimage import draw, io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import IPython
from collections import defaultdict


def random_crop(dims, img, mask, samples=100, box=200, side=10):
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
        rand_h = np.random.randint(0, h - 1 - box)
        rand_w = np.random.randint(0, w - 1 - box)

        crop_img = img[rand_h:rand_h + box, rand_w:rand_w + box]
        # io.imshow(crop_img)
        # plt.show()

        # TO DO apply cropped mask and variance threshold (NEED TO STUDY)
        # or keep within center of box with side modifications
        crop_mask = np.array(mask[rand_h + side:rand_h + box - side,
                                  rand_w + side:rand_w + box - side])

        whites = [0 if (pixel == [0, 0, 0, 255]).all()
                  else 1 for row in crop_mask for pixel in row]

        if sum(whites) > 0:
            print(sum(whites))
            io.imshow(crop_img)
            plt.show()

            io.imshow(crop_mask)
            plt.show()
        # print(np.unique(np.array(crop_mask), axis=1))

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
#     # selects pixel of top left corner
#     # np.random.randint(0, high=height)
# IPython.embed()

# print(row["mask"])
# mask = io.imread(row["mask"])
# print(mask[0][0])
# # print(mask[0] == [0, 0, 0, 255])

# # print(img)
# m_h, m_w = mask.shape[:2]
# i_h, i_w = img.shape[:2]
# # io.imshow(img)
# # plt.show()
# break
# if abs(m_h - i_h) > 0 or abs(m_w - i_w) > 0:
#     print(img.shape, mask.shape)

# mask = io.imread()
# print(data.columns)
# image = io.imread("/home/ariel/Dropbox/Training Data/26-10/img/page_0252.jpg")
# print(image.shape[:2])
