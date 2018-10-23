"""Convenience functs for processing labels"""

from collections import defaultdict
import json
import os
import pandas as pd
import re
import subprocess
import urllib.request


labels_dir = "./labels/"
master_hdf = lambda f: labels_dir + re.sub(r'\.json$', '', f) + ".hdf"


def create_dataframe(filename):
    """
    Create dataframe where each row contains location-based
    grouped handwritten elements, line-separated

    Assumes masks exported and retrieved with retrieve_masks()
    Might be better to integrate the two options

    Parameters
    ----------
    file: name of JSON file in ./labels that you wish to create dataframe for

    Returns
    ----------
    pandas dataframe in saved hdf in ./labels
    """

    with open(labels_dir + filename, "r", encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    elems = defaultdict(list)
    error = []

    for row in json_data:
        picid = row["External ID"]
        dataset = row["Dataset Name"].replace("(", "_").replace(")", "")
        items = row["Label"]
        masks = [] if "Masks" not in row else row["Masks"]
        path = "../data/%s/%s" % (dataset, picid)

        if "Text" in items:
            if "Start of text" in items:
                if len(items["Text"]) == len(items["Start of text"]):

                    mask_file = "../data/%s/text_mask/%s.png" % (
                        dataset, picid.split(".")[0])

                    for el, pt in zip(items["Text"], items["Start of text"]):
                        elems["hwType"].append("text")
                        elems["hasHR"].append("contains_handwriting" in items)
                        elems["pageid"].append(picid)
                        elems["path"].append(path)
                        elems["mask"].append(mask_file)
                        elems["mask_url"].append(masks["Text"])

                        if 'ease_in_reading' in el:
                            elems["readability"].append(
                                el['ease_in_reading'])
                        else:
                            error.append(("readability", row["View Label"]))

                        # TO DO vector direction implementation
                        # geo = elems["geometry"]
                        # elems["start_x"].append(pt["geometry"]["x"])
                        # elems["start_y"].append(pt["geometry"]["y"])

                        # TO DO add masks

                        # conditional elements
                        elems["isSig"].append("is_signature?" in el)
                        elems["isCrossed"].append("text_crossed-out" in el)
                        elems["isMarker"].append("was_marker?" in el)
                        elems["isFaint"].append("is_faint?" in el)
                        if "transcription" in el:
                            elems["transcript"].append(el["transcription"])
                        else:
                            elems["transcript"].append("")

                # every Text should have a Start of Text item
                else:
                    error.append(("start", row["View Label"]))
            # every Text should have a Start of Text item
            else:
                error.append(("start", row["View Label"]))

        if "Markings" in items:
            mask_file = "../data/%s/mark_mask/%s.png" % (
                dataset, picid.split(".")[0])

            for mark in items["Markings"]:
                elems["hwType"].append("mark")
                elems["hasHR"].append("contains_handwriting" in items)
                elems["pageid"].append(picid)
                elems["path"].append(path)
                elems["mask"].append(mask_file)
                elems["mask_url"].append(masks["Markings"])

                elems["readability"].append("")

                # TO DO when vector direction implementation
                # elems["start_x"].append(0)
                # elems["start_y"].append(0)

                # conditional elements
                elems["isSig"].append(False)
                elems["isCrossed"].append(False)
                elems["isMarker"].append(False)
                elems["isFaint"].append(False)
                elems["transcript"].append("")

        if "Text" not in items and "Markings" not in items:
            elems["hasHR"].append("contains_handwriting" in items)
            elems["pageid"].append(picid)
            elems["path"].append(path)
            elems["hwType"].append("")
            elems["readability"].append("")
            elems["mask"].append("")
            elems["mask_url"].append("")

            # TO DO when vector direction implementation
            # elems["start_x"].append(0)
            # elems["start_y"].append(0)

            # conditional elements
            elems["isSig"].append(False)
            elems["isCrossed"].append(False)
            elems["isMarker"].append(False)
            elems["isFaint"].append(False)
            elems["transcript"].append("")

    if len(error) > 0:
        print("[ERROR] check URLs in errors.txt and resolve issues:")
        file = open("errors.txt", "w")
        for cause, url in error:
            print("%s, \t\t%s\n" % (cause, url), file=file)
        file.close()
    else:
        df = pd.DataFrame(elems)
        return df.to_hdf(master_hdf(filename), "data")


def retrieve_masks(df):
    """
    Download all masks using wget to save in mask<pic#>_<mask#>.jpg

    Parameters
    ----------
    df : dataframe from create_dataframe

    Returns
    ----------
    saves df["mask_url"] to df["mask"] location

    Comments
    ----------
    might be able to improve with a sleep function between wget calls;
    instead of using urllib.

    """

    # reject where mask not set to value
    sel = df[df["mask"] != ""].copy()

    dirs = sel["mask"].apply(os.path.dirname)
    for out in dirs.unique():
        if len(out) > 1:
            subprocess.Popen(["mkdir", "-p", out])

    count = 0
    for loc, url in zip(sel["mask"], sel["mask_url"]):
        if os.path.isfile(loc):
            if os.path.getsize(loc) > 0:
                count += 1
                continue
            else:
                print(url)
                urllib.request.urlretrieve(url, loc)
        else:
            args = ['wget', '-O', loc, url]
            p = subprocess.Popen(args, stdout=subprocess.PIPE)

            # have to wait otherwise output scrambled
            os.waitpid(p.pid, 0)
            if os.path.isfile(loc) and os.path.getsize(loc) > 0:
                count += 1

    if count == sel.shape[0]:
        print("All specified masks have been downloaded. No need to re-run")
    else:
        print("Rerun retrieve_masks until receive message that all masks downloaded")


name = "22-10.json"
if False:  # create master dataframe
    create_dataframe(name)

if True:  # use master dateframe for selections, syncing data, etc.
    master = pd.read_hdf(master_hdf(name))

    # sync masks and images
    retrieve_masks(master)


# need out each element information of identified text and type
# need way to easily convert into hasHR or not
# to do: robustly check that no/yes keys correctly kept
# readme file
