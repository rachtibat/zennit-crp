import torch
import numpy as np
import gc
import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import re

from crp.concepts import Concept


class Maximization:

    def __init__(self, mode="relevance", max_target="sum", abs_norm=False, path=None):

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
            self.sub_folder = Path(f"RelMax_{max_target}_{norm_str}/")
        elif mode == "activation":
            self.sub_folder = Path(f"ActMax_{max_target}_{norm_str}/")
        else:
            raise ValueError("<mode> must be 'relevance' or 'activation'.")

        self.PATH = Path(path) / self.sub_folder if path else self.sub_folder

        self.PATH.mkdir(parents=True, exist_ok=True)
        # TODO: what happens if rf_c_sorted is empty? In sort and save
        # TODO: activation in save path instead of relevance!
        # TODO: for statistics in other class: make dummy variable for extra datset instead of SDS

    def analyze_layer(self, rel, concept: Concept, layer_name: str, data_indices, targets):

        b_c_sorted, rel_c_sorted, rf_c_sorted = concept.reference_sampling(
            rel, layer_name, self.max_target, self.abs_norm)
        # convert batch index to dataset wide index
        data_indices = torch.from_numpy(data_indices).to(b_c_sorted)
        d_c_sorted = torch.take(data_indices, b_c_sorted)
        # sort targets
        targets = torch.Tensor(targets).to(b_c_sorted)
        t_c_sorted = torch.take(targets, b_c_sorted)

        SZ = self.SAMPLE_SIZE
        self.concatenate_with_results(layer_name, d_c_sorted[:SZ], rel_c_sorted[:SZ], rf_c_sorted[:SZ])
        self.sort_result_array(layer_name)

        return d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted

    def delete_result_arrays(self):

        self.d_c_sorted, self.rel_c_sorted, self.rf_c_sorted = {}, {}, {}
        gc.collect()

    def concatenate_with_results(self, layer_name, d_c_sorted, rel_c_sorted, rf_c_sorted):

        if layer_name not in self.d_c_sorted:
            self.d_c_sorted[layer_name] = d_c_sorted
            self.rel_c_sorted[layer_name] = rel_c_sorted
            self.rf_c_sorted[layer_name] = rf_c_sorted

        else:
            self.d_c_sorted[layer_name] = torch.cat([d_c_sorted, self.d_c_sorted[layer_name]])
            self.rel_c_sorted[layer_name] = torch.cat([rel_c_sorted, self.rel_c_sorted[layer_name]])
            self.rf_c_sorted[layer_name] = torch.cat([rf_c_sorted, self.rf_c_sorted[layer_name]])

    def sort_result_array(self, layer_name):

        d_c_args = torch.flip(torch.argsort(self.rel_c_sorted[layer_name], dim=0), dims=(0,))
        d_c_args = d_c_args[:self.SAMPLE_SIZE, :]

        self.rel_c_sorted[layer_name] = torch.gather(self.rel_c_sorted[layer_name], 0, d_c_args)
        self.rf_c_sorted[layer_name] = torch.gather(self.rf_c_sorted[layer_name], 0, d_c_args)
        self.d_c_sorted[layer_name] = torch.gather(self.d_c_sorted[layer_name], 0, d_c_args)

    def _save_results(self, d_index: Tuple[int, int] = None):

        saved_files = []

        for layer_name in self.d_c_sorted:

            if d_index:
                filename = f"{layer_name}_{d_index[0]}_{d_index[1]}_"
            else:
                filename = f"{layer_name}_"

            np.save(self.PATH / Path(filename + "data.npy"), self.d_c_sorted[layer_name].cpu().numpy())
            np.save(self.PATH / Path(filename + "rf.npy"), self.rf_c_sorted[layer_name].cpu().numpy())
            np.save(self.PATH / Path(filename + "rel.npy"), self.rel_c_sorted[layer_name].cpu().numpy())

            saved_files.append(str(self.PATH / Path(filename)))

        self.delete_result_arrays()

        return saved_files

    def collect_results(self, path_list: List[str], d_index: Tuple[int, int] = None):

        self.delete_result_arrays()

        pbar = tqdm(total=len(path_list), dynamic_ncols=True)

        for path in path_list:
            filename = path.split("/")[-1]
            l_name = re.split(r"_[0-9]+_[0-9]+_\b", filename)[0]

            d_c_sorted = np.load(path + "data.npy")
            rf_c_sorted = np.load(path + "rf.npy")
            rel_c_sorted = np.load(path + "rel.npy")

            d_c_sorted, rf_c_sorted, rel_c_sorted = map(torch.from_numpy, [d_c_sorted, rf_c_sorted, rel_c_sorted])

            self.concatenate_with_results(l_name, d_c_sorted, rel_c_sorted, rf_c_sorted)
            self.sort_result_array(l_name)

            pbar.update(1)

        for path in path_list:
            for suffix in ["data.npy", "rf.npy", "rel.npy"]:
                os.remove(path + suffix)

        pbar.close()

        return self._save_results(d_index)
