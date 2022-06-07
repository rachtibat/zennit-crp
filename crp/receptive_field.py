import torch
import numpy as np
import math
from pathlib import Path
from typing import List, Dict

import warnings

from zennit.composites import *

from crp.attribution import CondAttribution
from crp.concepts import Concept
from crp.helper import get_output_shapes, load_receptive_field


class AllFlatComposite(LayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer
    '''

    def __init__(self, canonizers=None):
        layer_map = [
            (Linear, Flat()),
            (AvgPool, Flat()),
            (torch.nn.modules.pooling.MaxPool2d, Flat()),
            (Activation, Pass()),
            (Sum, Norm()),
        ]

        super().__init__(layer_map, canonizers=canonizers)


class ReceptiveField:

    def __init__(self, attribution: CondAttribution, single_sample: torch.tensor, path="FeatureVisualization"):
        """

        Parameter:
            single_sample: with batchsize 
        """

        self.rf_c_sorted = {}

        self.attribution = attribution
        self.single_sample = single_sample

        self.sub_folder = Path(f"ReField/")
        self.PATH = Path(path) / self.sub_folder if path else self.sub_folder

        self.PATH.mkdir(parents=True, exist_ok=True)

    def analyze_layer(self, concept: Concept, layer_name: str, c_indices, canonizer=None, batch_size=16, verbose=True):

        composite = AllFlatComposite(canonizer)
        conditions = [{layer_name: [index]} for index in c_indices]

        batch = 0
        for attr in self.attribution.generate(
                self.single_sample, conditions, composite, [], concept.mask_rf,
                layer_name, 1, batch_size, None, verbose):

            heat = self.norm_rf(attr.heatmap, layer_name)

            try:
                rf_array[batch * len(heat): (batch+1) * len(heat)] = heat
            except UnboundLocalError:
                rf_array = torch.zeros((len(c_indices), *heat.shape[1:]), dtype=torch.uint8)
                rf_array[batch * len(heat): (batch+1) * len(heat)] = heat

            batch += 1

        return rf_array

    def run(self, layer_map, canonizers: List = None, batch_size=16, verbose=True):

        print("Trace output shapes... ", end="")
        self.output_shapes = get_output_shapes(self.attribution.model, self.single_sample, list(layer_map.keys()))
        print("finished")

        saved_files = []
        for layer_name, concept in layer_map.items():
            print(f"Layer {layer_name}:")
            neuron_ids = concept.get_rf_indices(self.output_shapes[layer_name], layer_name)
            rf_layer = self.analyze_layer(concept, layer_name, neuron_ids, canonizers, batch_size, verbose)

            path = self.save_result(rf_layer, layer_name)
            saved_files.append(path)

        return saved_files

    def save_result(self, rf_layer, layer_name):

        path = self.PATH / Path(f"{layer_name}.npy")
        np.save(path, rf_layer.cpu().numpy())

        return str(path)

    def norm_rf(self, heatmaps, layer_name):
        """
        normalize between [0, 255]
        """

        result = torch.zeros_like(heatmaps, dtype=torch.uint8)
        heatmaps = torch.abs(heatmaps)
        for i, h in enumerate(heatmaps):

            if h.max() != 0:
                h = h / h.max()
                h = h * 255
            else:
                warnings.warn(f"Receptive field is zero for a neuron {i} in layer {layer_name}.")

            result[i] = h

        return result

    def weight_receptive_field(self, neuron_ids, tensor, layer_name):

        if len(neuron_ids) != len(tensor):
            raise ValueError("<neuron_ids> and <tensor> must be equal in length.")

        # magic number
        threshold = 40

        rf_array = load_receptive_field(self.PATH, layer_name)
        rf_array = torch.from_numpy(rf_array[neuron_ids]).to(tensor)

        images = []
        for i in range(len(neuron_ids)):
            rows, columns = torch.where(rf_array[i] > threshold)
            row1 = rows.min() if len(rows) != 0 else 0
            row2 = rows.max() if len(rows) != 0 else -1
            col1 = columns.min() if len(columns) != 0 else 0
            col2 = columns.max() if len(columns) != 0 else -1

            if (row1 < row2) and (col1 < col2):
                images.append(tensor[i, ..., row1:row2, col1:col2])

        return images
