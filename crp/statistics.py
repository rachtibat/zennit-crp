import torch
import numpy as np
import gc
import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

class Statistics:

    def __init__(self, mode="relevance", max_target="sum", abs_norm=False, path=None):

        self.d_c_sorted, self.rel_c_sorted, self.rf_c_sorted = {}, {}, {}
        self.SAMPLE_SIZE = 40

        # generate path string for filenames
        if abs_norm:
            norm_str = "normed"
        else:
            norm_str = "unnormed"

        if mode == "relevance":
            self.sub_folder = Path(f"RelStats_{max_target}_{norm_str}/")
        elif mode == "activation":
            self.sub_folder = Path(f"ActStats_{max_target}_{norm_str}/")
        else:
            raise ValueError("<mode> must be 'relevance' or 'activation'.")

        self.PATH = Path(path) / self.sub_folder if path else self.sub_folder

        self.PATH.mkdir(parents=True, exist_ok=True)
        # TODO: what happens if rf_c_sorted is empty? In sort and save method
        # TODO: activation in save path instead of relevance!

    def analyze_layer(self, d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted, layer_name):
        
        t_unique = torch.unique(t_c_sorted)

        for t in t_unique:

            # gather d_c, rel_c and rf_c for each target separately 
            t_indices = t_c_sorted.t() == t
            
            # - each column of t_c_sorted contains the same number of same value targets
            # - C-style arrays start indexing row-wise
            # - we transpose, so that reshaping the flattened array, that results of [t_indices] operation,
            #   maintains the order of elements
            n_concepts = t_c_sorted.shape[1]

            d_c_t = d_c_sorted.t()[t_indices].view(n_concepts, -1).t()
            rel_c_t = rel_c_sorted.t()[t_indices].view(n_concepts, -1).t()
            rf_c_t = rf_c_sorted.t()[t_indices].view(n_concepts, -1).t()

            self.concatenate_with_results(layer_name, t.item(), d_c_t, rel_c_t, rf_c_t)
            self.sort_result_array(layer_name, t.item())

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
            self.d_c_sorted[target][layer_name] = torch.cat([d_c_sorted, self.d_c_sorted[target][layer_name]])
            self.rel_c_sorted[target][layer_name] = torch.cat([rel_c_sorted, self.rel_c_sorted[target][layer_name]])
            self.rf_c_sorted[target][layer_name] = torch.cat([rf_c_sorted, self.rf_c_sorted[target][layer_name]])

    def sort_result_array(self, layer_name, target):

        d_c_args = torch.argsort(self.rel_c_sorted[target][layer_name], dim=0, descending=True)
        d_c_args = d_c_args[:self.SAMPLE_SIZE, :]

        self.rel_c_sorted[target][layer_name] = torch.gather(self.rel_c_sorted[target][layer_name], 0, d_c_args)
        self.rf_c_sorted[target][layer_name] = torch.gather(self.rf_c_sorted[target][layer_name], 0, d_c_args)
        self.d_c_sorted[target][layer_name] = torch.gather(self.d_c_sorted[target][layer_name], 0, d_c_args)

    def _save_results(self, d_index: Tuple[int, int] = None):

        saved_files = []

        for target in self.d_c_sorted:

            for layer_name in self.d_c_sorted[target]:

                if d_index:
                    filename = f"{target}_{d_index[0]}_{d_index[1]}_"
                else:
                    filename = f"{target}_"

                p_path = self.PATH / Path(layer_name)
                p_path.mkdir(parents=True, exist_ok=True)
               
                np.save(p_path / Path(filename + "data.npy"), self.d_c_sorted[target][layer_name].cpu().numpy())
                np.save(p_path / Path(filename + "rf.npy"), self.rf_c_sorted[target][layer_name].cpu().numpy())
                np.save(p_path / Path(filename + "rel.npy"), self.rel_c_sorted[target][layer_name].cpu().numpy())

                saved_files.append(str(p_path / Path(filename)))

        if d_index is None:
            # if final collection, then save targets
            np.save(self.PATH  / Path("targets.npy"), np.array(list(self.d_c_sorted.keys())))
        
        self.delete_result_arrays()

        return saved_files

    def collect_results(self, path_list: List[str], d_index: Tuple[int, int] = None):

        self.delete_result_arrays()

        pbar = tqdm(total=len(path_list), dynamic_ncols=True)

        for path in path_list:

            l_name, filename = path.split("/")[-2:]
            target = filename.split("_")[0]

            d_c_sorted = np.load(path + "data.npy")
            rf_c_sorted = np.load(path + "rf.npy")
            rel_c_sorted = np.load(path + "rel.npy")

            d_c_sorted, rf_c_sorted, rel_c_sorted = map(torch.from_numpy, [d_c_sorted, rf_c_sorted, rel_c_sorted])

            self.concatenate_with_results(l_name, target, d_c_sorted, rel_c_sorted, rf_c_sorted)
            self.sort_result_array(l_name, target)

            pbar.update(1)

        for path in path_list:
            for suffix in ["data.npy", "rf.npy", "rel.npy"]:
                os.remove(path + suffix)

        pbar.close()

        return self._save_results(d_index)
