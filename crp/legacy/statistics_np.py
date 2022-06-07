from matplotlib.pyplot import axis
import torch
import numpy as np
import math
import shutil
from pathlib import Path
from crp.concepts import Concept
import gc
from typing import List
import os

class Statistics:

    def __init__(self, mode="relevance", max_target="sum", abs_norm=False):

        self.d_c_sorted, self.rel_c_sorted, self.rf_c_sorted = {}, {}, {}
        self.SAMPLE_SIZE = 40

        # generate path string for filenames
        if abs_norm:
            norm_str = "normed"
        else:
            norm_str = "unnormed"

        if mode == "relevance":
            self.PATH = Path(f"RelStats_{max_target}_{norm_str}/")
        elif mode == "activation":
            self.PATH = Path(f"ActStats_{max_target}_{norm_str}/")

        self.PATH.mkdir(parents=True, exist_ok=True)
        #TODO: what happens if rf_c_sorted is empty? In sort and save
        #TODO: activation in save path instead of relevance!
        #TODO: for statistics in other class: make dummy variable for extra datset instead of SDS
        #TODO: how preprocessing?

    def analyze_layer(self, d_c_sorted, rel_c_sorted, rf_c_sorted, layer_name, targets):

        t_unique = np.unique(targets)
        for t in t_unique:

            t_indices = np.argwhere(targets == t).reshape(-1)

            d_c_t = d_c_sorted[t_indices]
            rel_c_t = rel_c_sorted[t_indices]
            rf_c_t = rf_c_sorted[t_indices]

            self.concatenate_with_results(layer_name, t, d_c_t, rel_c_t, rf_c_t)
            self.sort_result_array(layer_name, t)

    def delete_result_arrays(self):

        self.d_c_sorted, self.rel_c_sorted, self.rf_c_sorted = {}, {}, {}
        gc.collect()

    def concatenate_with_results(self, layer_name, target, d_c_sorted, rel_c_sorted, rf_c_sorted):

        if target not in self.d_c_sorted:
            self.d_c_sorted[target] = {}
            self.rel_c_sorted[target] = {}
            self.rf_c_sorted[target] = {}

        if layer_name not in self.d_c_sorted[target]:
            self.d_c_sorted[target][layer_name] = d_c_sorted
            self.rel_c_sorted[target][layer_name] = rel_c_sorted
            self.rf_c_sorted[target][layer_name] = rf_c_sorted

        else:
            self.d_c_sorted[target][layer_name] = np.concatenate([d_c_sorted, self.d_c_sorted[target][layer_name]])
            self.rel_c_sorted[target][layer_name] = np.concatenate([rel_c_sorted, self.rel_c_sorted[target][layer_name]])
            self.rf_c_sorted[target][layer_name] = np.concatenate([rf_c_sorted, self.rf_c_sorted[target][layer_name]])

    def sort_result_array(self, layer_name, target):

        d_c_args = np.flip(np.argsort(self.rel_c_sorted[target][layer_name], axis=0), axis=0)
        d_c_args = d_c_args[:self.SAMPLE_SIZE, :]

        self.rel_c_sorted[target][layer_name] = np.take_along_axis(self.rel_c_sorted[target][layer_name], d_c_args, axis=0)
        self.rf_c_sorted[target][layer_name] = np.take_along_axis(self.rf_c_sorted[target][layer_name], d_c_args, axis=0)
        self.d_c_sorted[target][layer_name] = np.take_along_axis(self.d_c_sorted[target][layer_name], d_c_args, axis=0)

    def save_results(self, data_start, data_end):

        saved_files = []

        for target in self.d_c_sorted:

            for layer_name in self.d_c_sorted[target]:

                filename = f"{target}_{layer_name}_{data_start}_{data_end}_"

                np.save(self.PATH  / Path(filename + "data.npy"), self.d_c_sorted[target][layer_name])
                np.save(self.PATH  / Path(filename + "rf.npy"), self.rf_c_sorted[target][layer_name])
                np.save(self.PATH  / Path(filename + "rel.npy"), self.rel_c_sorted[target][layer_name])

                saved_files.append(filename)

        self.delete_result_arrays()

        return saved_files

    def collect_results(self, data_start: int, data_end: int, list: List[str]):

        self.delete_result_arrays()
        
        n_samples = {}

        for filename in list:
            target, l_name, d_start, d_end, _ = filename.split("_")

            if target not in n_samples:
                n_samples[target] = {}
            if l_name not in n_samples[target]:
                n_samples[target][l_name] = 0

            n_samples[target][l_name] += (int(d_end) - int(d_start))

            d_c_sorted = np.load(self.PATH / Path(filename + "data.npy"))
            rf_c_sorted = np.load(self.PATH / Path(filename + "rf.npy"))
            rel_c_sorted = np.load(self.PATH / Path(filename + "rel.npy"))

            self.concatenate_with_results(l_name, target, d_c_sorted, rel_c_sorted, rf_c_sorted)
            self.sort_result_array(l_name, target)

        for target in n_samples:
            for l_name in n_samples[target]:
                if n_samples[target][l_name] != (data_end - data_start):
                    raise FileNotFoundError("Sorry, some files are missing. Please rerun this process.")

        for filename in list:
            for suffix in ["data.npy", "rf.npy", "rel.npy"]:
                os.remove(self.PATH / Path(filename + suffix))
    
        self.save_results(data_start, data_end)


