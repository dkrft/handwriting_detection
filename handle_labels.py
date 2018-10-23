"""Convenience functs for processing labels"""

from collections import defaultdict
import json
import pandas as pd
import re

labels_dir = "./labels/"


def retrieve_masks(json):
    """
    Download all masks using wget to save in mask<pic#>_<mask#>.jpg

    Parameters
    ----------
    json : json file from Labelbox

    Returns
    ----------
    files saved to ../data/masks


    """
    import os
    import subprocess

    save_dir = "../data/masks"
    subprocess.Popen("mkdir -p " + save_dir, shell=True, executable='/bin/bash')

    for row in json:
        ids = re.findall(r"\d+", row["External ID"])[0]
        masks = row["Masks"]
        for it, key in enumerate(masks.keys()):
            url = masks[key]

            # change if all one mask type
            # currently 0 would be well-aligned and 1 for difficult
            save_as = "mask%s_%s.png" % (ids, it)
            args = ['wget', '-O', '%s/%s' % (save_dir, save_as), url]
            p = subprocess.Popen(args, stdout=subprocess.PIPE)
            # makes so terminates in console but runs slower as checking each
            os.waitpid(p.pid, 0)
            print("%s retrieved" % save_as)


def create_dataframe(file):
    """
    Create dataframe where each row contains location-based
    grouped handwritten elements, line-separated

    Assumes masks exported and retrieved with retrieve_masks()
    Might be better to integrate the two options

    Parameters
    ----------
    json : json file

    Returns
    ----------
    pandas dataframe
    """

    with open(labels_dir + file, "r", encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    elems = defaultdict(list)
    # mask_dir = "../data/masks"

    error = []
    for row in json_data:
        picid = row["External ID"]
        dataset = row["Dataset Name"]
        items = row["Label"]
        path = "../data/%s/%s" % (dataset, picid)

        if "Text" in items:
            if "Start of text" in items:
                if len(items["Text"]) == len(items["Start of text"]):

                    for el, pt in zip(items["Text"], items["Start of text"]):
                        elems["hwType"].append("text")
                        elems["hasHR"].append("contains_handwriting" in items)
                        elems["pageid"].append(picid)
                        elems["path"].append(path)

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
            for mark in items["Markings"]:
                elems["hwType"].append("mark")
                elems["hasHR"].append("contains_handwriting" in items)
                elems["pageid"].append(picid)
                elems["path"].append(path)

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
        return df
        # df.to_hdf(labels_dir + re.sub(r'\.json$', '', file) + ".hdf", "data")

# retrieve_masks(json_data)

file = "22-10.json"
test = create_dataframe(file)
print(test[['hwType', 'readability', 'isSig', 'isCrossed',
            'isMarker', 'isFaint']])
print(sum(test.hwType == ""))


# need out each element information of identified text and type
# need way to easily convert into hasHR or not
# to do: separate fct to robustly check that no/yes keys correctly kept
