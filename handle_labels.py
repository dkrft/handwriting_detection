"""Convenience functs for processing labels"""

from collections import defaultdict
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import seaborn as sns
import subprocess
import urllib.request


labels_dir = "./labels/"
master_hdf = lambda f: labels_dir + re.sub(r'\.json$', '', f) + ".hdf"
sub_hdf = lambda f, m: labels_dir + \
    re.sub(r'\.json$', '', f) + "%s" % m + ".hdf"


def create_dataframe(filename):
    """
    Create dataframe where each row contains location-based
    grouped handwritten elements, line-separated

    Assumes masks exported and retrieved with retrieve_masks()
    Might be better to integrate the two options

    Parameters
    ----------
    filename: JSON file in ./labels that you wish to create dataframe for

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
        path = "../data/%s/img/%s" % (dataset, picid)

        if "Text" in items:
            if "Start of text" in items:
                if len(items["Text"]) == len(items["Start of text"]):

                    mask_file = "../data/%s/text_mask/%s.png" % (
                        dataset, picid.split(".")[0])

                    for el, pt in zip(items["Text"], items["Start of text"]):
                        elems["hwType"].append("text")
                        elems["hasHW"].append(1)
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
                        elems["start_x"].append(pt["geometry"]["x"])
                        elems["start_y"].append(pt["geometry"]["y"])

                        # conditional elements
                        elems["isSig"].append(int("is_signature?" in el))
                        elems["isCrossed"].append(int("text_crossed-out" in el))
                        elems["isMarker"].append(int("was_marker?" in el))
                        elems["isFaint"].append(int("is_faint?" in el))
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
                elems["hasHW"].append(1)
                elems["pageid"].append(picid)
                elems["path"].append(path)
                elems["mask"].append(mask_file)
                elems["mask_url"].append(masks["Markings"])

                elems["readability"].append("")

                # TO DO when vector direction implementation
                elems["start_x"].append(0)
                elems["start_y"].append(0)

                # conditional elements
                elems["isSig"].append(0)
                elems["isCrossed"].append(0)
                elems["isMarker"].append(0)
                elems["isFaint"].append(0)
                elems["transcript"].append("")

        if "Machine signature" in items:
            mask_file = "../data/%s/machine_mask/%s.png" % (
                dataset, picid.split(".")[0])

            for sig in items["Machine signature"]:
                elems["hwType"].append("mark")
                elems["hasHW"].append(0)
                elems["pageid"].append(picid)
                elems["path"].append(path)
                elems["mask"].append(mask_file)
                elems["mask_url"].append(masks["Machine signature"])

                elems["readability"].append("")

                # TO DO when vector direction implementation
                elems["start_x"].append(0)
                elems["start_y"].append(0)

                # conditional elements
                elems["isSig"].append(0)
                elems["isCrossed"].append(0)
                elems["isMarker"].append(0)
                elems["isFaint"].append(0)
                elems["transcript"].append("")

        if "Text" not in items and "Markings" not in items:
            elems["hasHW"].append(0)
            elems["pageid"].append(picid)
            elems["path"].append(path)
            elems["hwType"].append("")
            elems["readability"].append("")
            elems["mask"].append("")
            elems["mask_url"].append("")

            # TO DO when vector direction implementation
            elems["start_x"].append(0)
            elems["start_y"].append(0)

            # conditional elements
            elems["isSig"].append(0)
            elems["isCrossed"].append(0)
            elems["isMarker"].append(0)
            elems["isFaint"].append(0)
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
                urllib.request.urlretrieve(url, loc)
        else:
            args = ['wget', '-O', loc, url]
            p = subprocess.Popen(args, stdout=subprocess.PIPE)

            # have to wait otherwise output scrambled
            # could comment out if wanting to run faster & willing
            # to rerun retrieve_masks again with active
            os.waitpid(p.pid, 0)
            if os.path.isfile(loc) and os.path.getsize(loc) > 0:
                count += 1

    if count == sel.shape[0]:
        print("All specified masks have been downloaded. No need to re-run")
    else:
        print("Rerun retrieve_masks until receive message that all masks downloaded")


def hasHW_dataframe(df, filename):
    """
    Reduce master dataframe to dataframe with summary values and whether
    or not page has handwritten elements

    Parameters
    ----------
    df : dataframe from create_dataframe
    filename: JSON file in ./labels that you wish to create dataframe for

    Returns
    ----------
    pandas dataframe in saved hdf in ./labels
    """

    base = df.copy()

    aggs = {'hwType': lambda x: sum(x == "text"), "isSig": "sum",
            "isFaint": "max", "isMarker": "max", "isCrossed": "max"}
    renames = {'hwType': "numLines", "isSig": "numSigs", "isFaint": "hasFaint",
               "isMarker": "hasMarker", "isCrossed": "hasCrossed"}

    hasHW = base.groupby(["hasHW", "pageid", 'path'], as_index=False).agg(
        aggs).rename(columns=renames)

    hasHW.to_hdf(sub_hdf(filename, "_hasHW"), "data")


def stats(filename):
    """
    Generate statistics on HW elements from
    master_hdf and hasHW df

    Parameters
    ----------
    filename: original json string

    Returns
    ----------
    plots
    """
    df = pd.read_hdf(master_hdf(filename))
    hasHW = pd.read_hdf(sub_hdf(filename, "_hasHW"))

    # hist. line width

    # hist. line height

    # hist. no. characters

    # hist. no. lines per page

    # bar of types of marks

    # bar of readibility + signatures
    sns.countplot(x="readability", data=df)
    plt.show()


def uniform_randomizer(df_hasHW, level, samples=None, save_as="hdf"):
    """
    Create dataframe, folder, or whatever is needed for input into a NN
    or other ML system

    Parameters
    ----------
    df_hasHW : dataframe from hasHW_dataframe()

    level: 'all' : pages with a handwritten text and/or mark counted in HW set
           'text' : only pages with HW text counted in HW set; may have marks

           (not yet implemented)
           'only_text' : only pages with HW text counted in HW set; no marks

    samples: no. of pages of ea. no and (yes) handwritten elements
             would you like; if None, smaller dataset used as limiting factor

    save_as: "return": returns dataframe for use
             "hdf": saves HDF with 6-digit hash
             "folder": copies imgs to foler with 6-digit hash & creates hdf

    Returns
    ----------

    """
    import random
    import string

    HW_mask = {"all": (df_hasHW.hasHW == 1),
               "text": (df_hasHW.numLines > 0)}

    no_HW = df_hasHW[df_hasHW.hasHW == 0].copy()
    yes_HW = df_hasHW[HW_mask[level]].copy()

    # sampling performed without replacement
    def sampler(val):
        data = pd.concat([no_HW.sample(val), yes_HW.sample(val)])

        if save_as == "return":
            return data
        elif save_as == "hdf":
            file_id = ''.join(random.choices(
                string.ascii_uppercase + string.digits, k=6))
            # TO DO & add to file with specifications
            print("ok")
        elif save_as == "folder":
            print("ok")
            # TO DO & add to file with specifications
            # need to make sure names are unique...adding mapping

    # if requested sample larger than a population, no data produced
    if isinstance(samples, int) and samples > no_HW.shape[0]:
        print("[ERROR] Requested %s samples, but no HW set has %s pages" %
              (samples, no_HW.shape[0]))
    elif isinstance(samples, int) and samples > yes_HW.shape[0]:
        print("[ERROR] Requested %s samples, but HW set has %s pages" %
              (samples, yes_HW.shape[0]))
    else:
        # lots of checks. essentially whatever is small determines sample size
        if isinstance(samples, int):
            print("Sampling %s pages from each set." % samples)
            sampler(samples)

        elif no_HW.shape[0] <= yes_HW.shape[0]:
            print("Only %s in no HW set. Sampling from each set." %
                  no_HW.shape[0])
            sampler(no_HW.shape[0])
        else:
            print("Only %s in HW set. Sampling from each set." %
                  yes_HW.shape[0])
            sampler(yes_HW.shape[0])


# just change file name to new one in labels/ and switch to True block you
# want to run
name = "26-10.json"

if True:  # create master dataframe
    create_dataframe(name)

if False:  # use master dateframe for selections, syncing data, etc.
    master = pd.read_hdf(master_hdf(name))

    # sync masks and images
    retrieve_masks(master)

    # create hasHW dataframe
    hasHW_dataframe(master, name)

if False:  # use hasHW dataframe for selection into NN, etc.
    hasHW = pd.read_hdf(sub_hdf(name, "_hasHW"))
    uniform_randomizer(hasHW, "text", samples=50)


if False:  # generate histograms of data
    stats(name)

# to do: robustly check that no/yes keys correctly kept
