import torch
import numpy as np
from typing import List
import os
from pathlib import Path


def get_layer_names(model: torch.nn.Module, types: List):
    """
    Retrieves the layer names of all layers that belong to a torch.nn.Module type defined 
    in 'types'.

    Parameters
    ----------
    model: torch.nn.Module
    types: list of torch.nn.Module
        Layer types i.e. torch.nn.Conv2D

    Returns
    -------
    layer_names: list of strings


    """

    layer_names = []

    for name, layer in model.named_modules():
        for layer_definition in types:
            if isinstance(layer, layer_definition) or issubclass(layer.__class__, layer_definition):
                if name not in layer_names:
                    layer_names.append(name)

    return layer_names


def abs_norm(rel: torch.Tensor, stabilize=1e-10):
    """

    Parameter:
        rel: 1-D array
    """

    abs_sum = torch.sum(torch.abs(rel))

    return rel / (abs_sum + stabilize)

def max_norm(rel, stabilize=1e-10):
    
    return rel / (rel.max() + stabilize)


def get_output_shapes(model, single_sample: torch.tensor, record_layers: List[str]):
    """
    calculates the output shape of each layer using a forward pass.


    """

    output_shapes = {}

    def generate_hook(name):

        def shape_hook(module, input, output):
            output_shapes[name] = output.shape[1:]

        return shape_hook

    hooks = []
    for name, layer in model.named_modules():
        if name in record_layers:
            shape_hook = generate_hook(name)
            hooks.append(layer.register_forward_hook(shape_hook))

    _ = model(single_sample)

    [h.remove() for h in hooks]

    return output_shapes


def load_maximization(path_folder, layer_name):

    filename = f"{layer_name}_"

    d_c_sorted = np.load(Path(path_folder) / Path(filename + "data.npy"), mmap_mode="r")
    rel_c_sorted = np.load(Path(path_folder) / Path(filename + "rel.npy"), mmap_mode="r")
    rf_c_sorted = np.load(Path(path_folder) / Path(filename + "rf.npy"), mmap_mode="r")

    return d_c_sorted, rel_c_sorted, rf_c_sorted

def load_stat_targets(path_folder):

    targets = np.load(Path(path_folder) / Path("targets.npy")).astype(np.int)

    return targets


def load_statistics(path_folder, layer_name, target):

    filename = f"{target}_"

    d_c_sorted = np.load(Path(path_folder) / Path(layer_name) / Path(filename + "data.npy"), mmap_mode="r")
    rel_c_sorted = np.load(Path(path_folder) / Path(layer_name) / Path(filename + "rel.npy"), mmap_mode="r")
    rf_c_sorted = np.load(Path(path_folder) / Path(layer_name) / Path(filename + "rf.npy"), mmap_mode="r")

    return d_c_sorted, rel_c_sorted, rf_c_sorted


def load_receptive_field(path_folder, layer_name):

    filename = f"{layer_name}.npy"

    rf_array = np.load(Path(path_folder) / Path(filename), mmap_mode="r")

    return rf_array


def find_files(path=None):
    """
    Parameters:
        path: path analysis results

    """
    if path is None:
        path = os.getcwd()

    folders = os.listdir(path)

    r_max, a_max, r_stats, a_stats, rf = [], [], [], [], []
    for name in folders:
        found_path = str(Path(path) / Path(name))
        if "RelMax" in name:
            r_max.append(found_path)
        elif "ActMax" in name:
            a_max.append(found_path)
        elif "RelStats" in name:
            r_stats.append(found_path)
        elif "ActStats" in name:
            a_stats.append(found_path)
        elif "ReField" in name:
            rf.append(found_path)

    return r_max, a_max, r_stats, a_stats, rf
