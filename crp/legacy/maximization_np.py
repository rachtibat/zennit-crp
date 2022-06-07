import torch
import numpy as np
import math
from pathlib import Path
from crp.concepts import Concept
import gc
import os
from typing import List


class Maximization:

    def __init__(self, mode="relevance", max_target="sum", abs_norm=False):

        self.d_c_sorted, self.rel_c_sorted, self.rf_c_sorted = {}, {}, {}
        self.SAMPLE_SIZE = 40

        self.max_target = max_target
        self.abs_norm = abs_norm

        # generate path string for filenames
        if abs_norm:
            norm_str = "normed"
        else:
            norm_str = "unnormed"

        if mode == "relevance":
            self.PATH = Path(f"RelMax_{max_target}_{norm_str}/")
        elif mode == "activation":
            self.PATH = Path(f"ActMax_{max_target}_{norm_str}/")

        self.PATH.mkdir(parents=True, exist_ok=True)
        #TODO: what happens if rf_c_sorted is empty? In sort and save
        #TODO: activation in save path instead of relevance!
        #TODO: for statistics in other class: make dummy variable for extra datset instead of SDS
        #TODO: how preprocessing?

    def analyze_layer(self, rel, concept: Concept, layer_name, data_indices):

        d_c_sorted, rel_c_sorted, rf_c_sorted = concept.reference_sampling(rel, layer_name, self.max_target, self.abs_norm)
        # convert batch index to dataset wide index
        d_c_sorted = np.take(data_indices, d_c_sorted, axis=0)

        SZ = self.SAMPLE_SIZE
        self.concatenate_with_results(layer_name, d_c_sorted[:SZ], rel_c_sorted[:SZ], rf_c_sorted[:SZ])
        self.sort_result_array(layer_name)

        return d_c_sorted, rel_c_sorted, rf_c_sorted

    def delete_result_arrays(self):

        self.d_c_sorted, self.rel_c_sorted, self.rf_c_sorted = {}, {}, {}
        gc.collect()

    def concatenate_with_results(self, layer_name, d_c_sorted, rel_c_sorted, rf_c_sorted):

        if layer_name not in self.d_c_sorted:
            self.d_c_sorted[layer_name] = d_c_sorted
            self.rel_c_sorted[layer_name] = rel_c_sorted
            self.rf_c_sorted[layer_name] = rf_c_sorted

        else:
            self.d_c_sorted[layer_name] = np.concatenate([d_c_sorted, self.d_c_sorted[layer_name]])
            self.rel_c_sorted[layer_name] = np.concatenate([rel_c_sorted, self.rel_c_sorted[layer_name]])
            self.rf_c_sorted[layer_name] = np.concatenate([rf_c_sorted, self.rf_c_sorted[layer_name]])

    def sort_result_array(self, layer_name):

        d_c_args = np.flip(np.argsort(self.rel_c_sorted[layer_name], axis=0), axis=0)
        d_c_args = d_c_args[:self.SAMPLE_SIZE, :]

        self.rel_c_sorted[layer_name] = np.take_along_axis(self.rel_c_sorted[layer_name], d_c_args, axis=0)
        self.rf_c_sorted[layer_name] = np.take_along_axis(self.rf_c_sorted[layer_name], d_c_args, axis=0)
        self.d_c_sorted[layer_name] = np.take_along_axis(self.d_c_sorted[layer_name], d_c_args, axis=0) 


    def save_results(self, data_start, data_end):
        
        saved_files = []

        for layer_name in self.d_c_sorted:

            filename = f"{layer_name}_{data_start}_{data_end}_"

            np.save(self.PATH  / Path(filename + "data.npy"), self.d_c_sorted[layer_name])
            np.save(self.PATH  / Path(filename + "rf.npy"), self.rf_c_sorted[layer_name])
            np.save(self.PATH  / Path(filename + "rel.npy"), self.rel_c_sorted[layer_name])

            saved_files.append(filename)

        self.delete_result_arrays()

        return saved_files


    def collect_results(self, data_start: int, data_end: int, list: List[str]):

        self.delete_result_arrays()
        #TODO: kick out? because user might partition differently. 
        #n_samples = {}

        for filename in list:
            l_name, d_start, d_end, _ = filename.split("_")

      #      if l_name not in n_samples:
       #         n_samples[l_name] = 0
        #    n_samples[l_name] += (int(d_end) - int(d_start))

            d_c_sorted = np.load(self.PATH / Path(filename + "data.npy"))
            rf_c_sorted = np.load(self.PATH / Path(filename + "rf.npy"))
            rel_c_sorted = np.load(self.PATH / Path(filename + "rel.npy"))

            self.concatenate_with_results(l_name, d_c_sorted, rel_c_sorted, rf_c_sorted)
            self.sort_result_array(l_name)

       # for l_name in n_samples:
        #    if n_samples[l_name] != (data_end - data_start):
         #       raise FileNotFoundError("Sorry, some files are missing. Please rerun this process.")

        for filename in list:
            for suffix in ["data.npy", "rf.npy", "rel.npy"]:
                os.remove(self.PATH / Path(filename + suffix))
    
        self.save_results(data_start, data_end)