"""Use to sample pages for CNN input"""
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
# import IPython
from collections import defaultdict
import pickle

tuner = defaultdict(list)
save_dir = "../data/samples/"

pixel_maps = []
labels = []


def random_crops(pid, dims, img, masks="", samples=10, box=150, side=20,
                 keep_mach_sigs=0, save=0):
    """
    Create x random selections of a page with a fixed box size

    Parameters
    ----------
    dims : dimensions of the image being considered

    img: imread object of img (df["path"])

    masks: array with imread object of masks (df["mask"])

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
    it = 0

    while it < samples:
        # - bottom left pixel of box -
        rand_h = np.random.randint(0, h - box)
        rand_w = np.random.randint(0, w - box)

        crop_img = img[rand_h:rand_h + box, rand_w:rand_w + box]
        # has no mask or hw element
        if isinstance(masks, str):
            # if save:
            #     cv2.imwrite(save_dir + "no_mask/%s_%i.png" %
            #                 (pid, it), crop_img)
            # pixel_maps.append(crop_img)
            # labels.append(0)
            it += 1

        # page has mask(s)
        else:
            # TO DO apply cropped mask and variance threshold (NEED TO STUDY)
            # or keep within center of box with side modifications
            # TO TUNE FURTHER

            # this implementation would allow HW element near a machine element
            # does not distinguish between mark/text mixed cases
            for hwType, in_mask in masks.items():
                crop_mask = np.array(in_mask[rand_h:rand_h + box,
                                             rand_w:rand_w + box])

                whites = sum(crop_mask.flatten() == 255)

                # has indent to ensure HW element (label not on edge)
                sub_mask = np.array(in_mask[rand_h + side:rand_h + box - side,
                                            rand_w + side:rand_w + box - side])

                sub_whites = sum(sub_mask.flatten() == 255)

                if sub_whites > 0:
                    # to check on masking
                    # mask_img = cv2.bitwise_and(crop_img, crop_img,
                    #                            mask=crop_mask)
                    # plt.imshow(mask_img)
                    # plt.show()

                    if hwType == "mach_sig":

                        if keep_mach_sigs:
                            if save:
                                cv2.imwrite(save_dir + "mask/%s/%s_%i.png" %
                                            (hwType, pid, it), crop_img)
                            pixel_maps.append(crop_img)
                            labels.append(0)
                            it += 1
                            break

                        # don't want to save these; don't count this sample
                        else:
                            # continue as may have other masks it works with
                            continue

                    # assumed either text or mark; save out & new sample
                    else:
                        save_path = save_dir + "mask/%s/%s_%i.png" % \
                            (hwType, pid, it)

                        if save:
                            cv2.imwrite(save_path, crop_img)
                        pixel_maps.append(crop_img)
                        labels.append(1)

                        # tuner["hwType"].append(hwType)
                        # tuner["whites_sub_mask"].append(sub_whites)
                        # tuner["whites_mask"].append(whites)
                        # # tuner["nonblacks_mask_img"].append()

                        # # excludes mask
                        # mean, stdDev = cv2.meanStdDev(crop_img, mask=crop_mask)
                        # tuner["mean_r"].append(mean[0][0])
                        # tuner["mean_g"].append(mean[1][0])
                        # tuner["mean_b"].append(mean[2][0])

                        # tuner["std_r"].append(stdDev[0][0])
                        # tuner["std_g"].append(stdDev[1][0])
                        # tuner["std_b"].append(stdDev[2][0])

                        # # includes mask elements
                        # mean2, stdDev2 = cv2.meanStdDev(crop_img)

                        # tuner["mean_r_bkg"].append(mean2[0][0])
                        # tuner["mean_g_bkg"].append(mean2[1][0])
                        # tuner["mean_b_bkg"].append(mean2[2][0])

                        # tuner["std_r_bkg"].append(stdDev2[0][0])
                        # tuner["std_g_bkg"].append(stdDev2[1][0])
                        # tuner["std_b_bkg"].append(stdDev2[2][0])

                        # tuner["path"].append(save_path)

                        # tuner["pen"].append(input('mark y / n?'))
                        # tuner["char"].append(input('char(s) y / n?'))
                        # tuner["edge"].append(input('on edge y / n?'))

                        it += 1
                        # found a matching mask; no need to consider others
                        break

                # had mask but not included in crop
                else:
                    if save:
                        cv2.imwrite(save_dir + "mask/not_in_samp/%s_%i.png" %
                                    (pageid, it), crop_img)

                    pixel_maps.append(crop_img)
                    labels.append(0)

                    it += 1


data = pd.read_hdf("labels/26-10.hdf")

# reducing data to first element to only load img/mask once
sel = data.groupby(["pageid", "hwType"], as_index=False).first()

dims = defaultdict(list)

count = 0
for name, group in sel.groupby(["pageid", "path"], as_index=False):
    count += 1
    # if more than one mask, path will be duplicated
    path = group["path"].unique()[0]

    # creating unique name to save cropped image out to
    pageid = (re.sub("img/", "", path).split(".jpg")[0]).split("/")[2:4]
    pageid = "_".join(pageid)

    # 0 import in image and mask
    img = cv2.imread(path)
    h, w = img.shape[:2]
    dims["height"].append(h)
    dims["width"].append(w)

    scale_me = 1.
    if h < 2337 and w < 2337:
        if h > w:
            scale_me = 2337 / h
        else:
            scale_me = 2337 / w
    img = cv2.resize(img, (0, 0), fx=scale_me, fy=scale_me)
    h, w = img.shape[:2]

    # means that no masks are present; hasHW = 0
    if "" in group["mask"].unique():
        random_crops(pageid, (h, w), img)

    # has mask(s)
    else:
        masks = {}
        # need to iterate over for cropper
        for index, el in group.iterrows():
            # comes in inverted from saved png
            mask = cv2.imread(el["mask"], 0)
            mask = cv2.resize(mask, (0, 0), fx=scale_me, fy=scale_me)
            masks[el["hwType"]] = mask
        random_crops(pageid, (h, w), img, masks)

    if count % 20 == 0:
        print(path)

print(len(pixel_maps), len(labels))
if len(pixel_maps) == len(labels):
    f = open("random_26-10_10ea.pkl", "wb")
    pickle.dump(pixel_maps, f)
    pickle.dump(labels, f)
    f.close()

# samples = pd.DataFrame(tuner)
# samples.to_hdf("threshold.hdf", key="data")

# dimensions = pd.DataFrame(dims)
# dimensions.to_hdf("page_dims.hdf", key="data")

# TO DO | 2D histogram of page dimensions
