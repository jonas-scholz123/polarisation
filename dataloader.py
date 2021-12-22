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
        self.adjacency_matrix_path = f"{config.ADJACENCY_MATRIX_PATH}{config.MONTH}.csv"
        self.id_to_sub_path = f"{config.ID_DICT_PATH}{config.MONTH}.csv"
        self.edge_list_path = f"{config.EDGE_LIST_PATH}{config.MONTH}.csv"

        # Make folders if they don't exist.
        Path(config.ADJACENCY_MATRIX_PATH).mkdir(parents=True, exist_ok=True)
        Path(config.ID_DICT_PATH).mkdir(parents=True, exist_ok=True)

        # initialise results
        self.id_to_sub = None
        self.adjacency_matrix = None
        self.edge_list = None
        self.ensure_results_directories_exist()
    
    def ensure_results_directories_exist(self):
        self.ensure_dir_exists(config.EDGE_LIST_PATH)
        self.ensure_dir_exists(config.ADJACENCY_MATRIX_PATH)
        self.ensure_dir_exists(config.ID_DICT_PATH)

    def ensure_dir_exists(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def get_adjacency_matrix(self):
        # If already cached, no need to load. Simply return.
        if self.adjacency_matrix is None:
            # Else, either load from disk or rebuild.
            if config.REBUILD or not os.path.exists(self.adjacency_matrix_path):
                #self.network, self.id_to_sub = self.build_network_from_raw()
                network_builder.build_and_save_adjacency_matrix(
                    config.DATA_PATH,
                    self.adjacency_matrix_path,
                    self.id_to_sub_path,
                    config.MIN_COUNT_BOT_EXCLUSION,
                    config.SUBREDDIT_COMMENT_THRESHOLD)
            else:

                print("Loading data...")
                self.adjacency_matrix = pd.read_csv(self.adjacency_matrix_path).values
                print("Done")
        
        return self.adjacency_matrix
    
    def get_edge_list(self):
        # If already cached, no need to load. Simply return.
        if self.edge_list is None:
            # Else, either load from disk or rebuild.
            if config.REBUILD or not os.path.exists(self.edge_list_path):
                #self.network, self.id_to_sub = self.build_network_from_raw()
                network_builder.build_and_save_edge_list(
                    config.DATA_PATH,
                    self.edge_list_path,
                    self.id_to_sub_path,
                    config.MIN_COUNT_BOT_EXCLUSION,
                    config.SUBREDDIT_COMMENT_THRESHOLD)
            else:

                print("Loading data...")
                self.edge_list = pd.read_csv(self.edge_list_path)
                print("Done")
        
        return self.edge_list
    
    def get_id_to_sub(self):
        if self.id_to_sub is None:
            print("Loading data...")
            self.id_to_sub = pd.read_csv(self.id_to_sub_path).set_index("subreddit").to_dict()["id"]
            print("Done")
        return self.id_to_sub
# %%

dataloader = Dataloader()
a = dataloader.get_edge_list()
a.sort_values("weight", ascending=False)
# %%
