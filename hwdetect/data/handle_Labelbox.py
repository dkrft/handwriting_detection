"""Script for for processing the Labelbox JSON file

This script recursively goes through the elements of a JSON file containing
the labels and their reference images/masks created on Labelbox. For more
details on what is contained in it, see the about_json.txt

The original images are assumed to be locally stored inside a directory
called original_data:
    * img          contains original images
    * text_mask    will store masks associated with handwritten text
    * mark_mask    will store masks associated with non-textual handwriting
    * mach_mask    will store masks associated with machine-generated signatures
Masks are black and white images generated in Labelbox automatically, where
the white areas correspond to places where handwriting was identified.

In order to sample or select pages, a pandas dataframe is created and stored
in an HDF. The provided columns are described in the about_hdf.text file.

Usage
-------
Create HDF pandas dataframe of and masks for the PRImA data.
1. Set-up directory as described above.
2. Use commands given below with the path to the PRImA data and where you'd
   like the pandas HDF to be saved to.

>>> from hwdetect.data import handle_Labelbox
>>> handle_Labelbox.process_labelbox("../../data/labeled_databases/26-10.json")

* You may need to run this several times as not all masks may be retrieved;
we're not working with the paid-API front. You can just run retrieve_masks on
the HDF or the process_labelbox.

* Output if ran before on file
Pandas dataframe already created for 26-10.json
All specified masks have been downloaded. No need to re-run
Saved *hasHW.hdf to ../../data/labeled_databases/26-10_hasHW.hdf

"""


from collections import defaultdict
import json
import os
import pandas as pd
import re
import subprocess
import urllib.request

__author__ = "Ariel Bridgeman"
__version__ = "1.0"


base_savename = lambda json_path: re.sub(r'\.json$', '', json_path)


def create_dataframe(jsonfile, mask_path="../../data/original_data/"):
    """
    Create dataframe where each row contains location-based
    grouped handwritten elements, line-separated

    Assumes masks exported and retrieved with retrieve_masks()
    Might be better to integrate the two options

    Parameters
    ----------
    jsonfile : str
        full path to location of JSON file

    Outputs
    ----------
    HDF of pandas dataframe in saved hdf in ./labeled_databases

    Comments
    ----------
    For details about the JSON and HDF structure, see the about_json.txt
    and about_hdf.txt. Geometry (x, y bounding coordinates of objects) is not
    currently added as it would require special tricks (not all polygons
    have the same number of xys...and pandas not meant to hold dictionaries)

    """
    savepath = base_savename(jsonfile)

    # opening and loading contents of json file
    with open(jsonfile, "r", encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # creating holders to store data
    elems = defaultdict(list)
    error = []
    # iterating through each row of the json file, which is equal to 1
    # page (can contain several or just one classification)
    for row in json_data:
        picid = row["External ID"]
        dataset = row["Dataset Name"].replace("(", "_").replace(")", "")
        items = row["Label"]
        masks = [] if "Masks" not in row else row["Masks"]
        path = "../data/%s/img/%s" % (dataset, picid)

        if "Text" in items:
            if "Start of text" in items:
                if len(items["Text"]) == len(items["Start of text"]):

                    mask_file = mask_path + "/%s/text_mask/%s.png" % (
                        dataset, picid.split(".")[0])

                    for el, pt in zip(items["Text"], items["Start of text"]):
                        elems["hwType"].append("text")
                        elems["hasHW"].append(1)
                        elems["pageid"].append(picid)
                        elems["path"].append(path)
                        elems["mask"].append(mask_file)
                        elems["mask_url"].append(masks["Text"])
                        # elems["geometry"].append(el["geometry"])

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
            mask_file = mask_path + "/%s/mark_mask/%s.png" % (
                dataset, picid.split(".")[0])

            for mark in items["Markings"]:
                elems["hwType"].append("mark")
                elems["hasHW"].append(1)
                elems["pageid"].append(picid)
                elems["path"].append(path)
                elems["mask"].append(mask_file)
                elems["mask_url"].append(masks["Markings"])
                # elems["geometry"].append(mark["geometry"])

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
            mask_file = mask_path + "/%s/machine_mask/%s.png" % (
                dataset, picid.split(".")[0])

            for sig in items["Machine signature"]:
                elems["hwType"].append("mach_sig")
                elems["hasHW"].append(0)
                elems["pageid"].append(picid)
                elems["path"].append(path)
                elems["mask"].append(mask_file)
                elems["mask_url"].append(masks["Machine signature"])
                # elems["geometry"].append(sig["geometry"])

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
            # elems["geometry"].append({'x': 0, 'y': 0})

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
        df.to_hdf(savepath + ".hdf", "data")
        print("HDF saved to %s" % (savepath + ".hdf"))


def retrieve_masks(df):
    """
    Download all masks using wget to save in mask<pic#>_<mask#>.jpg

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        pandas dataframe from create_dataframe

    Output
    ----------
    saves mask as PNG from df["mask_url"] to df["mask"] location

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


def hasHW_dataframe(df, savepath):
    """
    Reduce master dataframe to dataframe with summary values and whether
    or not page has handwritten elements

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        pandase dataframe from create_dataframe
    savepath: str
        str path to save HDF

    Outputs
    ----------
    hdf
        pandas dataframe in saved hdf in same location as original JSON file
    """

    base = df.copy()

    aggs = {'hwType': lambda x: sum(x == "text"), "isSig": "sum",
            "isFaint": "max", "isMarker": "max", "isCrossed": "max"}
    renames = {'hwType': "numLines", "isSig": "numSigs", "isFaint": "hasFaint",
               "isMarker": "hasMarker", "isCrossed": "hasCrossed"}

    hasHW = base.groupby(["hasHW", "pageid", 'path'], as_index=False).agg(
        aggs).rename(columns=renames)

    hasHW.to_hdf(savepath + "_hasHW.hdf", "data")
    print("Saved *hasHW.hdf to %s" % (savepath + "_hasHW.hdf"))


def process_labelbox(jsonfile, mask_path="../../data/original_data/"):
    """Process all Labelbox JSON entries to generate pandas dataframes for
    quick look-ups and download masks for future use

    Parameters
    ----------
    jsonfile : str
        full path to location of JSON file
    mask_path : str
        path to directory where original_data is stored

    Outputs
    ----------
    png
        mask pngs saved to prima_dir + "/text_mask/"; generated
        from the coordinates given in the xmls
    hdf
        pandas dataframe HDF saved in same directory as JSON file

    """
    savepath = base_savename(jsonfile)

    # 1. create dataframe from JSON file
    if not os.path.isfile(savepath + ".hdf"):
        create_dataframe(jsonfile)
    else:
        print("\nPandas dataframe already created for %s" %
              os.path.basename(jsonfile))

    # load dataframe for next two steps
    df = pd.read_hdf(savepath + ".hdf", "data")

    # 2. retrieve masks as specified in df["mask"]
    retrieve_masks(df)

    # 3. save convenience hdf on page-level with having any handwriting or not
    hasHW_dataframe(df, savepath)
