import torch
import numpy as np
import math
from pathlib import Path
from typing import List, Union, Dict, Tuple

from crp.attribution import CondAttribution
from crp.maximization import Maximization
from crp.concepts import Concept
from crp.statistics import Statistics

import concurrent.futures


def run(dataset, nodes: int, worker: int, path, data_start=0, data_end=None):

    n_samples = data_end - data_start 

    if n_samples > nodes:
        batches = math.ceil(n_samples / nodes)
    else:
        batches = n_samples
        nodes = n_samples

    lines = []
    for b in range(batches):
        lines.append(f"")

        with open("crp_commands.txt", "w") as f:
            f.write

    


class FeatureVisualization:

    def __init__(self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept], device=None) -> None:
        
        self.dataset = dataset
        self.layer_map = layer_map

        self.attribution = attribution

        self.device = attribution.device if device is None else device

        self.MaxRel = Maximization("relevance")
        self.MaxAct = Maximization("activation")

        self.RelStats = Statistics("relevance")
        self.ActStats = Statistics("activation")

    def get_data_sample(self, index, preprocessing=True) -> Tuple[torch.tensor, int]:
        #TODO preprocessing delet?
        data, target = self.dataset[index]
        return data.to(self.device), target

    def multitarget_to_single(self, multi_target):
        
        raise NotImplementedError


    def run_analysis(self, composite, data_start, data_end, batch_size=32, checkpoint=500, on_device=None):
        """
        max batch_size = max(multi_targets) * data_batch
        data_end: exclusively counted
        """

        self.saved_checkpoints = {"max": [], "stats": []}
        last_checkpoint = 0
        
        n_samples = data_end - data_start 
        samples = np.arange(start=data_start, stop=data_end)

        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        if composite:
            composite.register(self.attribution.model)

        for b in range(batches):
            print(f"Run Zennit on Sample Batch {b + 1}/{batches}")

            samples_batch = samples[b * batch_size: (b + 1) * batch_size]

            data_batch, targets_samples = self.get_data_concurrently(samples_batch, preprocessing=True)
            targets_samples = np.array(targets_samples) # numpy operation needed

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
                data_broadcast = torch.stack(data_broadcast, dim=0)
                sample_indices = np.array(sample_indices)
                
            except NotImplementedError:
                data_broadcast, targets, sample_indices = data_batch, targets_samples, samples_batch

            conditions = {self.attribution.MODEL_OUTPUT_NAME: targets}
            record_layer = [l_name for l_name in iter(self.layer_map.keys())]

            _, activations, relevances = self.attribution(data_broadcast, conditions, None, record_layer)

            for layer_name, concept in self.layer_map.items():

                act_l, rel_l = activations[layer_name], relevances[layer_name]

                self.analyze_layer(act_l, rel_l, layer_name, concept, sample_indices, targets)

            if b % checkpoint == checkpoint -1:
                self.save_results(last_checkpoint, sample_indices[-1] + 1)
                last_checkpoint = sample_indices[-1] + 1
        
        #TODO: what happens if result arrays are empty?
        self.save_results(last_checkpoint, sample_indices[-1] + 1)
        composite.remove()

        return self.saved_checkpoints

    @torch.no_grad()
    def analyze_layer(self, act, rel, layer_name, concept, data_indices, targets):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """
        #TODO: dummy target for extra dataset
        d_c_sorted, rel_c_sorted, rf_c_sorted = self.MaxRel.analyze_layer(rel, concept, layer_name, data_indices)

        self.RelStats.analyze_layer(d_c_sorted, rel_c_sorted, rf_c_sorted, layer_name, targets)

        # activation analysis once per sample if multi target dataset
        unique_indices = np.unique(data_indices, return_index=True)[1]
        data_indices = data_indices[unique_indices]
        act = act[unique_indices]
        targets = targets[unique_indices]

        d_c_sorted, act_c_sorted, rf_c_sorted = self.MaxAct.analyze_layer(act, concept, layer_name, data_indices)
        
        self.ActStats.analyze_layer(d_c_sorted, act_c_sorted, rf_c_sorted, layer_name, targets)

    def save_results(self, start, end):

        self.saved_checkpoints["max"].extend(self.MaxRel._save_results(start, end))
        self.MaxAct._save_results(start, end)
        self.saved_checkpoints["stats"].extend(self.RelStats._save_results(start, end))
        self.ActStats._save_results(start, end)

    def collect_results(self, data_start: int, data_end: int, checkpoints: Dict[str, List[str]]):

        saved_files = {"max" : [], "stats": []}
        saved_files["max"].extend(self.MaxAct.collect_results(data_start, data_end, checkpoints["max"]))
        self.MaxRel.collect_results(data_start, data_end, checkpoints["max"])
        saved_files["stats"].extend(self.RelStats.collect_results(data_start, data_end, checkpoints["stats"]))
        self.ActStats.collect_results(data_start, data_end, checkpoints["stats"])

        return saved_files

    #TODO: write dataloader because of multi target
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

        data_returned = torch.stack(data_returned, dim=0)
        return data_returned, labels_returned


