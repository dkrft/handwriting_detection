"""Convenience functs for processing labels"""

from collections import defaultdict
import json
import pandas as pd
import re

jsons_dir = "./labels/"
with open(jsons_dir + "first50_20_10.json", "r", encoding='utf-8') as json_file:
    json_data = json.load(json_file)


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


def create_pandas(json):
    """
    Create dataframe where each row contains location-based
    grouped handwritten elements; not line-separated yet

    Parameters
    ----------
    json : json file

    Returns
    ----------
    df: pandas dataframe
    """
    holder = defaultdict(list)

    for row in json:
        picid = row["External ID"]

        items = row["Label"]
        for key in items:
            # rejects scan info which is just clear/fuzzy/etc.
            if not isinstance(items[key], str):
                for it in items[key]:
                    # repeated elements
                    holder["pic"] = picid

                    # need to change for multi-color text if needed
                    color = it['select_text_color']
                    holder["color"] = color[0] if len(
                        color) == 1 else "multi-color"

                    holder["reading_ease"] = it[
                        'how_easy_is_it_to_read_the_handwriting?']
                    holder["el_type"] = it['type_of_handwriting_element']
                    holder["meaning"] = it[
                        'what_does_the_selected_text_element_say?']
                    holder["words"] = it[
                        'what_is_the_orientation_of_the_text_within_the_shape?']
                    # print(it)
            # if isinstance(items[key], dict):
            #
            # print(it)
#         # print(, row["Masks"].keys())
#         # print()
#         # for sub in row["Label"]:
#         #     holder["pic"] = row["External ID"]
#         #     print(sub)
#         #     print(row["Label"])
#         #     break

# retrieve_masks(json_data)
create_pandas(json_data)
