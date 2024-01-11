import requests
import gzip
import yaml
import shutil
import json
import pandas as pd
import os

with open("./config/config.yaml", "rt") as config_file:
    config = yaml.safe_load(config_file)


class DataLoader:
    def __init__(self):
        pass


    def download_and_extract_from_url(self,url,save_to = None):
        r = requests.get(url, stream=True)
        if not save_to:
            save_to = url.split("/")[-1].split(".gz")[0]
        save_to = "./data/raw/"+save_to
        if r.status_code == 200:
            with open(save_to, 'wb') as f:
                r.raw.decode_content = True  # just in case transport encoding was applied
                gzip_file = gzip.GzipFile(fileobj=r.raw)
                shutil.copyfileobj(gzip_file, f)


    def load_raw_as_df(self):
        data = []
        raw_files = os.listdir("./data/raw/")
        # print(raw_files)
        for file in raw_files:
            with open("./data/raw/"+file,'r') as file:
                for line in file:
                    data.append(json.loads(line))
        df = pd.DataFrame(data)
        return df
    



