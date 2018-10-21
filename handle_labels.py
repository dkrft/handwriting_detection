"""Convenience functs for processing labels"""

import json
import pandas as pd

jsons_dir = "./labels/"
with open(jsons_dir + "first50_20_10.json", "r", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
