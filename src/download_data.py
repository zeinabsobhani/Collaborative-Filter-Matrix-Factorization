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
    """
    Class responsible for downloading and handling raw data.
    """
    def __init__(self):
        pass

    def download_and_extract_from_url(self,url:str = None,save_to:str = None):
        """
        Download and decompress and save .gz file given the url into the data/raw directory.
        Args:
            url (str): The url to download the data from. If not specified will use the url provided by the config file.
            save_to (str): the filename to save as. If not specified, will use the name in the url.            
        """
        if not url:
            url = config['data_url']
        if not save_to:
            save_to = url.split("/")[-1].split(".gz")[0]
        save_to = "./data/raw/"+save_to

        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(save_to, 'wb') as f:
                r.raw.decode_content = True  # just in case transport encoding was applied
                gzip_file = gzip.GzipFile(fileobj=r.raw)
                shutil.copyfileobj(gzip_file, f)


    def load_raw_as_df(self):
        """
        Load all the json files in data/raw into pandas dataframe.
        Returns:
            df (pd.DataFrame): dataframe of raw data
        """
        data = []
        raw_files = os.listdir("./data/raw/")
        for file in raw_files:
            with open("./data/raw/"+file,'r') as file:
                for line in file:
                    data.append(json.loads(line))
        df = pd.DataFrame(data)
        return df
    

