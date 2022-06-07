from socket import IOCTL_VM_SOCKETS_GET_LOCAL_CID
from matplotlib import table
import torch
import numpy as np
import math
from pathlib import Path
from typing import List, Union, Dict, Tuple
import os
from crp.attribution import CondAttribution
from crp.maximization import Maximization
from crp.concepts import Concept
from crp.statistics import Statistics
from crp.hooks import FeatVisHook
from crp.helper import load_maximization, load_statistics, load_stat_targets
from tqdm import tqdm
from crp.receptive_field import ReceptiveField

from typing import Callable
from zennit.composites import NameMapComposite, Composite
import zennit.image as zimage
import concurrent.futures


class FeatureVisualization:

    def __init__(
            self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept], preprocess_fn: Callable=None,
            max_target="sum", abs_norm=True, path="FeatureVisualization", device=None):

        self.dataset = dataset
        self.layer_map = layer_map
        self.preprocess_fn = preprocess_fn

        self.attribution = attribution

        self.device = attribution.device if device is None else device

        self.RelMax = Maximization("relevance", max_target, abs_norm, path)
        self.ActMax = Maximization("activation", max_target, abs_norm, path)

        self.RelStats = Statistics("relevance", max_target, abs_norm, path)
        self.ActStats = Statistics("activation", max_target, abs_norm, path)

        self.ReField = None

    def preprocess_data(self, data):

        if callable(self.preprocess_fn):
            return self.preprocess_fn(data)
        else:
            return data

    def get_data_sample(self, index, preprocessing=True) -> Tuple[torch.tensor, int]:
        """
        returns a data sample from dataset at index.

        Parameter:
            index: integer
            preprocessing: boolean.
                If True, return the sample after preprocessing. If False, return the sample for plotting.
        """

        data, target = self.dataset[index]
        data = data.to(self.device).unsqueeze(0)
        if preprocessing:
            data = self.preprocess_data(data)
        
        data.requires_grad = True
        return data, target

    def multitarget_to_single(self, multi_target):

        raise NotImplementedError

    def run(self, composite: Composite, data_start, data_end, batch_size=32, checkpoint=500, on_device=None):

        print("Running Analysis...")
        saved_checkpoints = self.run_distributed(composite, data_start, data_end, batch_size, checkpoint, on_device)

        print("Collecting results...")
        saved_files = self.collect_results(saved_checkpoints)

        return saved_files

    def run_distributed(self, composite: Composite, data_start, data_end, batch_size=16, checkpoint=500, on_device=None):
        """
        max batch_size = max(multi_targets) * data_batch
        data_end: exclusively counted
        """

        self.saved_checkpoints = {"r_max": [], "a_max": [], "r_stats": [], "a_stats": []}
        last_checkpoint = 0

        n_samples = data_end - data_start
        samples = np.arange(start=data_start, stop=data_end)

        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        # feature visualization is performed inside forward and backward hook of layers
        name_map, dict_inputs = [], {}
        for l_name, concept in self.layer_map.items():
            hook = FeatVisHook(self, concept, l_name, dict_inputs, on_device)
            name_map.append(([l_name], hook))
        fv_composite = NameMapComposite(name_map)

        if composite:
            composite.register(self.attribution.model)
        fv_composite.register(self.attribution.model)

        pbar = tqdm(total=batches, dynamic_ncols=True)

        for b in range(batches):

            pbar.update(1)

            samples_batch = samples[b * batch_size: (b + 1) * batch_size]
            data_batch, targets_samples = self.get_data_concurrently(samples_batch, preprocessing=True)

            targets_samples = np.array(targets_samples)  # numpy operation needed

            # convert multi target to single target if user defined the method
            data_broadcast, targets, sample_indices = [], [], []
            try:
                for i_t, target in enumerate(targets_samples):
                    single_targets = self.multitarget_to_single(target)
                    for st in single_targets:
                        targets.append(st)
                        data_broadcast.append(data_batch[i_t])
                        sample_indices.append(samples_batch[i_t])
                if len(data_broadcast) == 0:
                    continue
                # TODO: test stack
                data_broadcast = torch.stack(data_broadcast, dim=0)
                sample_indices = np.array(sample_indices)

            except NotImplementedError:
                data_broadcast, targets, sample_indices = data_batch, targets_samples, samples_batch

            conditions = [{self.attribution.MODEL_OUTPUT_NAME: [t]} for t in targets]
            # dict_inputs is linked to FeatHooks
            dict_inputs["sample_indices"] = sample_indices
            dict_inputs["targets"] = targets

            self.attribution(data_broadcast, conditions, None)

            if b % checkpoint == checkpoint - 1:
                self._save_results((last_checkpoint, sample_indices[-1] + 1))
                last_checkpoint = sample_indices[-1] + 1

        # TODO: what happens if result arrays are empty?
        self._save_results((last_checkpoint, sample_indices[-1] + 1))

        if composite:
            composite.remove()
        fv_composite.remove()

        pbar.close()

        return self.saved_checkpoints

    @torch.no_grad()
    def analyze_relevance(self, rel, layer_name, concept, data_indices, targets):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """
        # TODO: dummy target for extra dataset
        d_c_sorted, rel_c_sorted, rf_c_sorted = self.RelMax.analyze_layer(rel, concept, layer_name, data_indices)

        self.RelStats.analyze_layer(d_c_sorted, rel_c_sorted, rf_c_sorted, layer_name, targets)

    @torch.no_grad()
    def analyze_activation(self, act, layer_name, concept, data_indices, targets):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """

        # activation analysis once per sample if multi target dataset
        unique_indices = np.unique(data_indices, return_index=True)[1]
        data_indices = data_indices[unique_indices]
        act = act[unique_indices]
        targets = targets[unique_indices]

        d_c_sorted, act_c_sorted, rf_c_sorted = self.ActMax.analyze_layer(act, concept, layer_name, data_indices)

        self.ActStats.analyze_layer(d_c_sorted, act_c_sorted, rf_c_sorted, layer_name, targets)

    def _save_results(self, d_index=None):

        self.saved_checkpoints["r_max"].extend(self.RelMax._save_results(d_index))
        self.saved_checkpoints["a_max"].extend(self.ActMax._save_results(d_index))
        self.saved_checkpoints["r_stats"].extend(self.RelStats._save_results(d_index))
        self.saved_checkpoints["a_stats"].extend(self.ActStats._save_results(d_index))

    def collect_results(self, checkpoints: Dict[str, List[str]], d_index: Tuple[int, int] = None):

        saved_files = {}

        saved_files["r_max"] = self.RelMax.collect_results(checkpoints["r_max"], d_index)
        saved_files["a_max"] = self.ActMax.collect_results(checkpoints["a_max"], d_index)
        saved_files["r_stats"] = self.RelStats.collect_results(checkpoints["r_stats"], d_index)
        saved_files["a_stats"] = self.ActStats.collect_results(checkpoints["a_stats"], d_index)

        return saved_files

    # TODO: write dataloader because of multi target and speed
    def get_data_concurrently(self, indices: Union[List, np.ndarray, torch.tensor], preprocessing=False):

        if len(indices) == 1:
            data, label = self.get_data_sample(indices[0], preprocessing)
            return data, label

        threads = []
        data_returned = []
        labels_returned = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for index in indices:
                future = executor.submit(self.get_data_sample, index, preprocessing)
                threads.append(future)

        for t in threads:
            single_data = t.result()[0]
            single_label = t.result()[1]
            data_returned.append(single_data)
            labels_returned.append(single_label)

        data_returned = torch.cat(data_returned, dim=0)
        return data_returned, labels_returned

    def add_receptive_field(self, RF: ReceptiveField):

        self.ReField = RF

    def get_max_reference(
            self, concept_ids: list, layer_name: str, mode="relevance", r_range: Tuple[int, int] = (0, 8),
            heatmap=False, composite=None, batch_size=32, rf=True):
        """
        Retreive reference samples for a list of concepts in a layer. Relevance and Activation Maximization
        are availble if FeatureVisualization was computed for the mode. If the ReceptiveField of the layer
        was computed, it can be used to cut out the most representative part of the sample. In addition,
        conditional heatmaps can be computed on reference samples.

        Parameters:
        ----------
            concept_ids: list
            layer_name: str
            mode: "relevance" or "activation"
                Relevance or Activation Maximization 
            r_range: Tuple(int, int)
                Aange of N-top reference samples. For example, (3, 7) corresponds to the Top-3 to -6 samples.
                Argument must be a closed set i.e. second element of tuple > first element.
            heatmap: boolean
                If True, compute conditional heatmaps on reference samples. Please make sure to supply a composite.
            composite: zennit.composites or None
                If `heatmap` is True, `composite` is used for the CondAttribution object.
            batch_size: int
                If heatmap is True, describes maximal batch size of samples to compute for conditional heatmaps.
            rf: boolean
                If True, crop samples or heatmaps with receptive field using the `weight_receptive_field` method.

        Returns:
        -------
            ref_c: dictionary.
                Key values correspond to channel index and values are reference samples.
                If rf is True, reference samples are a list of torch.Tensor with different shapes. Otherwise the 
                dictionary values are torch.Tensor with same shape.
        """

        ref_c = {}

        if mode == "relevance":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.RelMax.PATH, layer_name)
        elif mode == "activation":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.ActMax.PATH, layer_name)
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        for c_id in concept_ids:

            d_indices = d_c_sorted[r_range[0]:r_range[1], c_id]

            if heatmap:
                data_batch, _ = self.get_data_concurrently(d_indices, preprocessing=True)
                data_batch = self.attribution_on_reference(data_batch, c_id, layer_name, composite, batch_size)
            else:
                data_batch, _ = self.get_data_concurrently(d_indices, preprocessing=False)

            if rf and self.ReField:
                neuron_ids = rf_c_sorted[r_range[0]:r_range[1], c_id]
                data_batch = self.ReField.weight_receptive_field(neuron_ids, data_batch, layer_name)

            ref_c[c_id] = data_batch

        return ref_c

    def attribution_on_reference(self, data, concept_id: int, layer_name: str, composite, batch_size=32):
        
        n_samples = len(data)
        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        heatmaps = []
        for b in range(batches):
            data_batch = data[b * batch_size: (b + 1) * batch_size]

            conditions = [{layer_name: [concept_id]}] 
            attr = self.attribution(data_batch, conditions, composite, start_layer=layer_name)

            heatmaps.append(attr.heatmap)

        return torch.cat(heatmaps, dim=0)

    def compute_stats(self, concept_id, layer_name: str, mode="relevance", top_N=5, mean_N=10, norm=False):

        if mode == "relevance":
            path = self.RelStats.PATH
        elif mode == "activation":
            path = self.ActStats.PATH 
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")
        
        targets = load_stat_targets(path)

        rel_target = torch.zeros(len(targets))
        for i, t in enumerate(targets):
            _, rel_c_sorted, _ = load_statistics(path, layer_name, t)
            rel_target[i] = float(rel_c_sorted[:mean_N, concept_id].mean())
        
        
        args = torch.argsort(rel_target, descending=True)[:top_N]

        sorted_t = targets[args]
        sorted_rel = rel_target[args]

        if norm:
            sorted_rel = sorted_rel / sorted_rel[0]
        
        return sorted_t, sorted_rel

    def get_stats_reference( self, concept_id: int, layer_name: str, targets: list, mode="relevance", r_range: Tuple[int, int] = (0, 8),
            heatmap=False, composite=None, batch_size=32, rf=True):

        ref_t = {}

        if mode == "relevance":
            path = self.RelStats.PATH
        elif mode == "activation":
            path = self.ActStats.PATH 
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        for t in targets:
            
            d_c_sorted, _, rf_c_sorted = load_statistics(path, layer_name, t)
            d_indices = d_c_sorted[r_range[0]:r_range[1], concept_id]

            if heatmap:
                data_batch, _ = self.get_data_concurrently(d_indices, preprocessing=True)
                data_batch = self.attribution_on_reference(data_batch, concept_id, layer_name, composite, batch_size)
            else:
                data_batch, _ = self.get_data_concurrently(d_indices, preprocessing=False)

            if rf and self.ReField:
                neuron_ids = rf_c_sorted[r_range[0]:r_range[1], concept_id]
                data_batch = self.ReField.weight_receptive_field(neuron_ids, data_batch, layer_name)

            ref_t[t] = data_batch

        return ref_t