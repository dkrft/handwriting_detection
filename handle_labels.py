"""Convenience functs for processing labels"""

from collections import defaultdict
import json
# import pandas as pd
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


def create_pandas(json):
    """
    Create dataframe where each row contains location-based
    grouped handwritten elements; not line-separated yet

    Assumes masks exported and retrieved with retrieve_masks()
    Might be better to integrate the two options

    Parameters
    ----------
    json : json file

    Returns
    ----------
    pandas dataframe
    """
    page = defaultdict(list)
    elems = defaultdict(list)
    mask_dir = "../data/masks"

    error = []
    for row in json:
        picid = row["External ID"]
        dataset = row["Dataset Name"]
        items = row["Label"]
        path = "../data/%s/%s" % (dataset, picid)

        # per page easy, # elems of handwritten, # elems marks
        page["hasHR"].append("contains_handwriting" in items)
        page["pageid"].append(picid)
        page["path"].append(path)

        if "Text" in items:
            if "Start of text" in items:
                text_el = items["Text"]
                point_el = items["Start of text"]
                if len(text_el) == len(point_el):
                    for el in items["Text"]:
                        elems[""]
                        elems["is_sig"].append("is_signature?" in el)
                        elems["readability"].append(el['ease_in_reading'])

                # every Text should have a Start of Text item
                else:
                    error.append(row["View Label"])

            # every Text should have a Start of Text item
            else:
                error.append(row["View Label"])

        if len(error) > 0:
            print("[ERROR] check URLs for missing Start of Text elements:")
            for err in error:
                print(err)
            print("\nAfter resolving, export and try again.")
            return
        else:
            return  # dataframe

        # from Text
        # page["num_hw"] =
        # page["num_sig"] =
        # page["num_faint"] =

        # from Markings
        # page["num_marks"] =

    #         # rejects scan info which is just clear/fuzzy/etc.
    #         # if just 1-classification could be simplified
    #         if not isinstance(items[key], str):

    #             for it in items[key]:
    #                 # repeated elements
    #                 holder["pic"].append(picid)

    #                 save_as = "mask%s_%s.png" % (re.findall(r"\d+", picid)[0],
    #                                              0 if key == "Well-aligned"
    #                                              else 1)
    #                 holder["mask"].append('%s/%s' % (mask_dir, save_as))

    #                 # need to change for multi-color text if needed
    #                 color = it['select_text_color']
    #                 holder["color"].append(color[0] if len(
    #                     color) == 1 else "multi-color")

    #                 holder["reading_ease"].append(
    #                     it['how_easy_is_it_to_read_the_handwriting?'])
    #                 holder["el_type"].append(it['type_of_handwriting_element'])
    #                 holder["meaning"].append(
    #                     it['what_does_the_selected_text_say?'])
    #                 holder["words"].append(
    #                     it['what_is_the_orientation_of_the_text_within_the_shape?'])

    # df = pd.DataFrame(holder)
    # df.to_hdf(labels_dir + re.sub(r'\.json$', '', file) + ".hdf", "data")

# retrieve_masks(json_data)

file = "22-10.json"
with open(labels_dir + file, "r", encoding='utf-8') as json_file:
    json_data = json.load(json_file)

create_pandas(json_data)


# need out each element information of identified text and type
# need way to easily convert into hasHR or not
