import torch
import numpy as np
import math
from pathlib import Path
from typing import List, Dict
from zennit.composites import *

from crp.attribution import CondAttribution
from crp.concepts import Concept

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

    def __init__(
            self, attribution: CondAttribution, single_sample: torch.tensor, mode="relevance", max_target="sum",
            abs_norm=False):

        self.rf_c_sorted = {}

        self.attribution = attribution
        self.single_sample = single_sample

        self.max_target = max_target
        self.abs_norm = abs_norm

        # generate path string for filenames
        if abs_norm:
            norm_str = "normed"
        else:
            norm_str = "unnormed"

        self.PATH = Path(f"ReField_{max_target}_{norm_str}/")

        self.PATH.mkdir(parents=True, exist_ok=True)

    def analyze_layer(self, concept: Concept, layer_name: str, c_indices, canonizer=None, batch_size=32):
        
        composite = AllFlatComposite(canonizer)
        conditions = {layer_name: c_indices}
        rf_array = []
        for attr, _, _ in self.attribution.generate(
                self.single_sample, conditions, concept.mask_rf, composite, [],
                layer_name, batch_size, False):

            rf_array.append(attr)

        return torch.cat(rf_array, dim=0)

    def analyze_files(self, layer_map: Dict[str, Concept], path_list: List[str]):

        l_file_dict = {}
        for path in path_list:
                
            filename = path.split("/")[-1]
            l_name = filename.split("_")[0]

            if l_name not in l_file_dict:
                l_file_dict[l_name] = []
            l_file_dict[l_name].append(path)


        for layer_name, concept in layer_map.items():

            files = l_file_dict[layer_name]
            rf_c = []
            for path in files:
                rf_c_part = np.load(path + "rf.npy")
                rf_c.append(rf_c_part)
            rf_c = np.concatenate(rf_c, axis=0).reshape(-1)

            neuron_ids = np.unique(rf_c)
            rf_layer = self.analyze_layer(concept, layer_name, neuron_ids)

            self.save_results(rf_layer, neuron_ids, layer_name)



    def norm_rf_heatmaps(self, heatmaps):

        # normalize between 0 and 255
        result = np.zeros_like(heatmaps, dtype=np.uint8)

        for i, r in enumerate(heatmaps):

            if r.max() != 0:
                r = r / r.max()
                r = r * 255
            else:
                warnings.warn("Receptive field is for one neuron zero.")

            result[i] = r

        return result

    

    def collect_results(self, command_arguments):
        """
        Checks whether all neurons where analyzed. If so, concatenate the result for every layer.
        """
        print("Check if all files were calculated...")
        files = []

        # get list of all files that should be calculated
        for command in command_arguments:

            layer_start, neuron_start, layer_end, neuron_end = self.command_to_parameters(command)

            for i_l in range(layer_start, layer_end + 1):  # +1 to include layer_end

                _, _, neuron_start_layer, neuron_end_layer = self.get_neuron_indices_layer(i_l, layer_start, layer_end,
                                                                                           neuron_start,
                                                                                           neuron_end)

                files.append([i_l, neuron_start_layer, neuron_end_layer])

                # check if file exists

                file_name = Path(f"{i_l}_{neuron_start_layer}_{neuron_end_layer}.p")
                file_path = (self.save_path_tmp / file_name)

                if not file_path.exists():
                    print(f"At least file: {i_l}_{neuron_start_layer}_{neuron_end_layer}.p missing")
                    return -1

        print("All files completed! Start collecting ...")

        k = 0  # index of last concatenated files
        for i in range(len(files)):

            if i == len(files) - 1 or files[i + 1][0] != files[i][0]:
                # last file or next file in queue is for another layer
                # concatenate all files from one layer
                rf_tmp = np.array([])
                files_to_delete = []
                for q in range(k, i + 1):
                    file_name = f"{files[q][0]}_{files[q][1]}_{files[q][2]}.p"
                    files_to_delete.append(file_name)
                    loaded_file = loadFile(self.save_path_tmp, file_name)

                    rf_tmp = np.concatenate((rf_tmp, loaded_file)) if len(rf_tmp) > 0 else loaded_file

                k = i + 1

                layer_name = list(self.MG.named_modules.keys())[files[i][0]]
                t_path = self.save_path / f"layer_{layer_name}.npy"
                np.save(t_path, rf_tmp)

                # save rf_indices to neuron_indices mapping, as neuron index 0 is not at index 0 at rf!
                neuron_indices, _ = self.get_to_analyze_neurons(layer_name)
                rf_index = np.arange(0, len(rf_tmp))
                rf_to_neuron_index = dict(zip(neuron_indices, rf_index))

                t_path = self.save_path / f"layer_{layer_name}_indices.npy"
                # np.save(t_path, rf_to_neuron_index.astype(np.uint16))  #TODO: delete line

                saveFile(self.save_path, f"layer_{layer_name}_indices.p", rf_to_neuron_index)

                print(f"Layer_{layer_name} collected")

        shutil.rmtree(self.save_path_tmp)
