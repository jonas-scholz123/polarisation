# %%
import pandas as pd
import numpy as np
import os
import network_builder
from tqdm import tqdm
from pprint import pprint
import pickle
from pathlib import Path
import gc

import config

class Dataloader:

    def __init__(self):
        self.network_path = f"{config.NETWORK_PATH}{config.MONTH}.pkl"
        self.id_to_sub_path = f"{config.ID_DICT_PATH}{config.MONTH}.pkl"

        # Make folders if they don't exist.
        Path(config.NETWORK_PATH).mkdir(parents=True, exist_ok=True)
        Path(config.ID_DICT_PATH).mkdir(parents=True, exist_ok=True)

        # initialise results
        self.id_to_sub = None
        self.network = None
    
    def build_network_from_raw(self):
        print("Loading raw data...")
        # data is split into multiple csvs. Want to merge and clean.
        files = [config.DATA_PATH + fname for fname in os.listdir(config.DATA_PATH)]
        if config.SAMPLE_DF:
            df = pd.read_csv(files[0], sep=";")
        else:
            df = pd.concat([pd.read_csv(fpath, sep=";") for fpath in files])
        
        print("Done!")
        print("Cleaning data & applying filters...")

        df = df.rename(columns={config.COUNT_COL_NAME : "count"})

        authors = df["author"]

        # we drop all deleted comments, because we can't trace them
        # to a specific user.
        # We drop comments made by the AutoModerator bot
        # And comments made by accounts that are likely bots.

        to_drop = authors.isin({"[deleted]", "AutoModerator"}) \
            ^ (authors.str.lower().str.endswith("bot") \
                & df["count"] >= config.MIN_COUNT_BOT_EXCLUSION)

        df = df[~to_drop]

        total_subreddit_comments = df.groupby("subreddit")["count"].sum().to_dict()

        subs_below_threshold = []
        for sub, count in sorted(total_subreddit_comments.items(), key = lambda x: x[1]):
            subs_below_threshold.append(sub)
            if count > config.SUBREDDIT_COMMENT_THRESHOLD:
                break
        subs_below_threshold = set(subs_below_threshold)
        df = df[~df["subreddit"].isin(subs_below_threshold)]
        gc.collect()

        df["subreddit"] = df["subreddit"].astype("category")
        id_to_sub = dict(enumerate(df["subreddit"].cat.categories))
        print("Done!")
        print("Building network...")
        overlaps = network_builder.build_network(df["author"], df["subreddit"].cat.codes, df["count"])

        return np.array(overlaps), id_to_sub
    
    def get_network(self):
        # If already cached, no need to load. Simply return.
        if self.network is None:
            # Else, either load from disk or rebuild.
            if config.REBUILD or not os.path.exists(self.network_path):
                self.network, self.id_to_sub = self.build_network_from_raw()
                if config.SAVE:
                    self.save_network()
            else:
                self.network, self.id_to_sub = self.load_network_from_file()
        
        return self.network, self.id_to_sub
    
    def save_network(self):
        with open(f"{config.NETWORK_PATH}{config.MONTH}.pkl", "wb") as f:
            pickle.dump(self.network, f)
        with open(f"{config.ID_DICT_PATH}{config.MONTH}.pkl", "wb") as f:
            pickle.dump(self.id_to_sub, f)
    
    def load_network_from_file(self):
        print("Loading data...")
        with open(self.network_path, "rb") as f:
            overlaps = pickle.load(f)

        with open(self.id_to_sub_path, "rb") as f:
            id_to_sub = pickle.load(f)
        print("Done!")

        return overlaps, id_to_sub
