"""Convenience functs for processing labels"""

from collections import defaultdict
import json
import pandas as pd
import re

jsons_dir = "./labels/"
with open(jsons_dir + "first50_20_10.json", "r", encoding='utf-8') as json_file:
    json_data = json.load(json_file)


def retrieve_masks(json):
    """Download all masks"""
    import subprocess

    save_dir = "../data/masks"
    subprocess.Popen("mkdir -p " + save_dir, shell=True, executable='/bin/bash')

    for row in json:
        ids = re.findall(r"\d+", row["External ID"])[0]
        masks = row["Masks"]
        for it, key in enumerate(masks.keys()):
            url = masks[key]

            # change if all one mask type
            save_as = "mask%s_%s.jpg" % (ids, it)
            args = ['wget', '-O', '%s/%s' % (save_dir, save_as), url]
            subprocess.Popen(' '.join(args), shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)


# def create_pandas(json):
#     """ Create dataframe where each row contains 1 mask
#     Input
#     --------
#     json   json file
#     """
#     holder = defaultdict(list)

#     for row in json:
#         if row["External ID"] == "page0005.jpg":
#             print(row["Label"].keys())
#         # print(row["External ID"], row["Masks"].keys())
#         # print()
#         # for sub in row["Label"]:
#         #     holder["pic"] = row["External ID"]
#         #     print(sub)
#         #     print(row["Label"])
#         #     break
# create_pandas(json_data)
retrieve_masks(json_data)
