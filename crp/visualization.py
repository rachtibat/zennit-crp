from typing import List, Union, Dict, Tuple, Callable
import warnings
import torch
import numpy as np
import math
from collections.abc import Iterable
import concurrent.futures
import functools
import inspect
from tqdm import tqdm
from zennit.composites import NameMapComposite, Composite
from crp.attribution import CondAttribution
from crp.maximization import Maximization
from crp.concepts import ChannelConcept, Concept
from crp.statistics import Statistics
from crp.hooks import FeatVisHook
from crp.helper import load_maximization, load_statistics, load_stat_targets
from crp.image import vis_img_heatmap, vis_opaque_img
from crp.cache import Cache


class FeatureVisualization:

    def __init__(
            self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept], preprocess_fn: Callable=None,
            max_target="sum", abs_norm=True, path="FeatureVisualization", device=None, cache: Cache=None):

        self.dataset = dataset
        self.layer_map = layer_map
        self.preprocess_fn = preprocess_fn

        self.attribution = attribution

        self.device = attribution.device if device is None else device

        self.RelMax = Maximization("relevance", max_target, abs_norm, path)
        self.ActMax = Maximization("activation", max_target, abs_norm, path)

        self.RelStats = Statistics("relevance", max_target, abs_norm, path)
        self.ActStats = Statistics("activation", max_target, abs_norm, path)

        self.Cache = cache

    def preprocess_data(self, data):

        if callable(self.preprocess_fn):
            return self.preprocess_fn(data)
        else:
            return data

    def get_data_sample(self, index, preprocessing=True) -> Tuple[torch.Tensor, int]:
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
                targets = np.array(targets)

            except NotImplementedError:
                data_broadcast, targets, sample_indices = data_batch, targets_samples, samples_batch

            conditions = [{self.attribution.MODEL_OUTPUT_NAME: [t]} for t in targets]
            # dict_inputs is linked to FeatHooks
            dict_inputs["sample_indices"] = sample_indices
            dict_inputs["targets"] = targets

            # composites are already registered before
            self.attribution(data_broadcast, conditions, None, exclude_parallel=False)

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
        d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted = self.RelMax.analyze_layer(
            rel, concept, layer_name, data_indices, targets)

        self.RelStats.analyze_layer(d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted, layer_name)

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

        d_c_sorted, act_c_sorted, rf_c_sorted, t_c_sorted = self.ActMax.analyze_layer(
            act, concept, layer_name, data_indices, targets)

        self.ActStats.analyze_layer(d_c_sorted, act_c_sorted, rf_c_sorted, t_c_sorted, layer_name)

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


    def cache_reference(func):
        """
        Decorator for get_max_reference and get_stats_reference. If a crp.cache object is supplied to the FeatureVisualization object,
        reference samples are cached i.e. saved after computing a visualization with a 'plot_fn' (argument of get_max_reference) or
        loaded from the disk if available.
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            """
            Parameters:
            -----------
            overwrite: boolean
                If set to True, already computed reference samples are computed again (overwritten).
            """
            
            overwrite = kwargs.pop("overwrite", False)
            args_f = inspect.getcallargs(func, self, *args, **kwargs)
            plot_fn = args_f["plot_fn"]

            if self.Cache is None or plot_fn is None:
                return func(**args_f)

            r_range, mode, l_name, rf, composite = args_f["r_range"], args_f["mode"], args_f["layer_name"], args_f["rf"], args_f["composite"]
            f_name, plot_name = func.__name__, plot_fn.__name__
            if f_name == "get_max_reference":
                indices = args_f["concept_ids"]
            else:
                indices = [f'{args_f["concept_id"]}:{i}' for i in args_f["targets"]]

            if overwrite:
                not_found = {id: r_range for id in indices}
                ref_c = {}
            else:
                ref_c, not_found = self.Cache.load(indices, l_name, mode, r_range, composite, rf, f_name, plot_name)

            if len(not_found):
                
                for id in not_found:
                    
                    args_f["r_range"] = not_found[id]

                    if f_name == "get_max_reference":
                        args_f["concept_ids"]  = id
                        ref_c_left = func(**args_f)
                    elif f_name == "get_stats_reference":
                        args_f["targets"] = int(id.split(":")[-1])
                        ref_c_left = func(**args_f)
                    else:
                        raise ValueError("Only the methods 'get_max_reference' and 'get_stats_reference' can be decorated.")

                    self.Cache.save(ref_c_left, l_name, mode, not_found[id], composite, rf, f_name, plot_name)

                    ref_c = self.Cache.extend_dict(ref_c, ref_c_left)

            return ref_c

        return wrapper

    @cache_reference
    def get_max_reference(
            self, concept_ids: Union[int,list], layer_name: str, mode="relevance", r_range: Tuple[int, int] = (0, 8), composite: Composite=None,
            rf=False, plot_fn=vis_img_heatmap, batch_size=32)-> Dict:
        """
        Retreive reference samples for a list of concepts in a layer. Relevance and Activation Maximization
        are availble if FeatureVisualization was computed for the mode. In addition, conditional heatmaps can be computed on reference samples.
        If the crp.concept class (supplied to the FeatureVisualization layer_map) implements masking for a single neuron in the 'mask_rf' method, 
        the reference samples and heatmaps can be cropped using the receptive field of the most relevant or active neuron.

        Parameters:
        ----------
        concept_ids: int or list
        layer_name: str
        mode: "relevance" or "activation"
            Relevance or Activation Maximization 
        r_range: Tuple(int, int)
            Range of N-top reference samples. For example, (3, 7) corresponds to the Top-3 to -6 samples.
            Argument must be a closed set i.e. second element of tuple > first element.
        composite: zennit.composites or None
            If set, compute conditional heatmaps on reference samples. `composite` is used for the CondAttribution object.
        rf: boolean
            If True, compute the CRP heatmap for the most relevant/most activating neuron only to restrict the conditonal heatmap
            on the receptive field.
        plot_fn: callable function with signature (samples: torch.Tensor, heatmaps: torch.Tensor, rf: boolean) or None
            Draws reference images. The function receives as input the samples used for computing heatmaps before preprocessing 
            with self.preprocess_data and the final heatmaps after computation. In addition, the boolean flag 'rf' is passed to it.
            The return value of the function should correspond to the Cache supplied to the FeatureVisualization object (if available).
            If None, the raw tensors are returned.
        batch_size: int
            If heatmap is True, describes maximal batch size of samples to compute for conditional heatmaps.

        Returns:
        -------
        ref_c: dictionary.
            Key values correspond to channel index and values are reference samples. The values depend on the implementation of
            the 'plot_fn'.
        """

        ref_c = {}
        if not isinstance(concept_ids, Iterable):
            concept_ids = [concept_ids]
        if mode == "relevance":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.RelMax.PATH, layer_name)
        elif mode == "activation":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.ActMax.PATH, layer_name)
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        if rf and not composite:
            warnings.warn("The receptive field is only computed, if you fill the 'composite' argument with a zennit Composite.")

        for c_id in concept_ids:

            d_indices = d_c_sorted[r_range[0]:r_range[1], c_id]
            n_indices = rf_c_sorted[r_range[0]:r_range[1], c_id]

            ref_c[c_id] = self._load_ref_and_attribution(d_indices, c_id, n_indices, layer_name, composite, rf, plot_fn, batch_size)

        return ref_c

    @cache_reference
    def get_stats_reference(self, concept_id: int, layer_name: str, targets: Union[int, list], mode="relevance", r_range: Tuple[int, int] = (0, 8),
            composite=None, rf=False, plot_fn=vis_img_heatmap, batch_size=32):
        """
        Retreive reference samples for a single concept in a layer wrt. different explanation targets i.e. returns the reference samples
        that are computed by self.compute_stats. Relevance and Activation are availble if FeatureVisualization was computed for the statitics mode. 
        In addition, conditional heatmaps can be computed on reference samples. If the crp.concept class (supplied to the FeatureVisualization layer_map) 
        implements masking for a single neuron in the 'mask_rf' method, the reference samples and heatmaps can be cropped using the receptive field of 
        the most relevant or active neuron.

        Parameters:
        ----------
        concept_ids: int or list
        layer_name: str
        mode: "relevance" or "activation"
            Relevance or Activation Maximization 
        r_range: Tuple(int, int)
            Range of N-top reference samples. For example, (3, 7) corresponds to the Top-3 to -6 samples.
            Argument must be a closed set i.e. second element of tuple > first element.
        composite: zennit.composites or None
            If set, compute conditional heatmaps on reference samples. `composite` is used for the CondAttribution object.
        rf: boolean
            If True, compute the CRP heatmap for the most relevant/most activating neuron only to restrict the conditonal heatmap
            on the receptive field.
        plot_fn: callable function with signature (samples: torch.Tensor, heatmaps: torch.Tensor, rf: boolean)
            Draws reference images. The function receives as input the samples used for computing heatmaps before preprocessing 
            with self.preprocess and the final heatmaps after computation. In addition, the boolean flag 'rf' is passed to it.
            The return value of the function should correspond to the Cache supplied to the FeatureVisualization object (if available).
            If None, the raw tensors are returned.
        batch_size: int
            If heatmap is True, describes maximal batch size of samples to compute for conditional heatmaps.

        Returns:
        -------
        ref_t: dictionary.
            Key values correspond to target indices and values are reference samples. The values depend on the implementation of
            the 'plot_fn'.
        """
            
        
        ref_t = {}
        if not isinstance(targets, Iterable):
            targets = [targets]
        if mode == "relevance":
            path = self.RelStats.PATH
        elif mode == "activation":
            path = self.ActStats.PATH 
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")
        
        if rf and not composite:
            warnings.warn("The receptive field is only computed, if you fill the 'composite' argument with a zennit Composite.")

        for t in targets:
            
            d_c_sorted, _, rf_c_sorted = load_statistics(path, layer_name, t)
            d_indices = d_c_sorted[r_range[0]:r_range[1], concept_id]
            n_indices = rf_c_sorted[r_range[0]:r_range[1], concept_id]

            ref_t[f"{concept_id}:{t}"] = self._load_ref_and_attribution(d_indices, concept_id, n_indices, layer_name, composite, rf, plot_fn, batch_size)

        return ref_t

    def _load_ref_and_attribution(self, d_indices, c_id, n_indices, layer_name, composite, rf, plot_fn, batch_size):

        data_batch, _ = self.get_data_concurrently(d_indices, preprocessing=False)

        if composite:
            data_p = self.preprocess_data(data_batch)
            heatmaps = self._attribution_on_reference(data_p, c_id, layer_name, composite, rf, n_indices, batch_size)

            if callable(plot_fn):
                return plot_fn(data_batch.detach(), heatmaps.detach(), rf)
            else:
                return data_batch.detach().cpu(), heatmaps.detach().cpu()

        else:
            return data_batch.detach().cpu()

    def _attribution_on_reference(self, data, concept_id: int, layer_name: str, composite, rf=False, neuron_ids: list=[], batch_size=32):

        n_samples = len(data)
        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        if rf and (len(neuron_ids) != n_samples):
            raise ValueError("length of 'neuron_ids' must be equal to the length of 'data'")

        heatmaps = []
        for b in range(batches):
            data_batch = data[b * batch_size: (b + 1) * batch_size].detach().requires_grad_()
            
            if rf:
                neuron_ids = neuron_ids[b * batch_size: (b + 1) * batch_size]
                conditions = [{layer_name: {concept_id: n_index}} for n_index in neuron_ids]
                attr = self.attribution(data_batch, conditions, composite, mask_map=ChannelConcept.mask_rf, start_layer=layer_name, on_device=self.device, 
                    exclude_parallel=False)
            else:
                conditions = [{layer_name: [concept_id]}] 
                # initialize relevance with activation before non-linearity (could be changed in a future release)
                attr = self.attribution(data_batch, conditions, composite, start_layer=layer_name, on_device=self.device, exclude_parallel=False)

            heatmaps.append(attr.heatmap)

        return torch.cat(heatmaps, dim=0)

    def compute_stats(self, concept_id, layer_name: str, mode="relevance", top_N=5, mean_N=10, norm=False) -> Tuple[list, list]:
        """
        Computes statistics about the targets i.e. classes that are most relevant or most activating for the concept with index 'concept_id'
        in layer 'layer_name'. Statistics must be computed before utilizing this method.

        Parameters:
        -----------
        concept_id: int
            Index of concept
        layer_name: str
        mode: str, 'relevance' or 'activation'
        top_N: int
            Returns the 'top_N' classes that most activate or are most relevant for the concept.
        mean_N: int
            Computes the importance of each target using the 'mean_N' top reference images for each target.
        norm: boolean
            If True, returns the mean relevance for each target normed.

        Returns:
        --------
        sorted_t, sorted_val as tuple
        sorted_t: list of most relevant targets
        sorted_val: list of respective mean relevance/activation values for each target 
        """

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
        sorted_val = rel_target[args]

        if norm:
            sorted_val = sorted_val / sorted_val[0]
        
        return sorted_t, sorted_val

    def _save_precomputed(self, s_tensor, h_tensor, index, plot_list, layer_name, mode, r_range, composite, rf, f_name):

        for plot_fn in plot_list:
            ref = {index: plot_fn(s_tensor, h_tensor, rf)}
            self.Cache.save(ref, layer_name, mode, r_range, composite, rf, f_name, plot_fn.__name__)

    def precompute_ref(self, layer_c_ind:Dict[str, List], composite: Composite, rf=True, stats=False, top_N=4, mean_N=10, mode="relevance", r_range: Tuple[int, int] = (0, 8), plot_list=[vis_opaque_img], batch_size=32):
        """
        Precomputes and saves all reference samples resulting from 'self.get_ref_samples' and 'self.get_stats_reference' for concepts supplied in 'layer_c_ind'.

        Parameters:
        -----------
        layer_c_ind: dict with str keys and list values
            Keys correspond to layer names and values to a list of all concept indices
        stats: boolean
            If True, precomputes reference samples of 'self.get_stats_reference'. Otherwise, only samples of 'self.get_ref_samples' are computed.
        plot_list: list of callable functions
            Functions to plot and save the images. The signature should correspond to the 'plot_fn' of 'get_max_reference'.

        REMAINING PARAMETERS: correspond to 'self.get_ref_samples' and 'self.get_stats_reference'
        """


        if self.Cache is None:
            raise ValueError("You must supply a crp.Cache object to the 'FeatureVisualization' class to precompute reference images!")
        
        if composite is None:
            raise ValueError("You must supply a zennit.Composite object to precompute reference images!")

        for l_name in layer_c_ind:

            c_indices = layer_c_ind[l_name]
            print("Layer:", l_name)
            pbar = tqdm(total=len(c_indices), dynamic_ncols=True)

            for c_id in c_indices:

                s_tensor, h_tensor = self.get_max_reference(c_id, l_name, mode, r_range, composite, rf, None, batch_size)[c_id]

                self._save_precomputed(s_tensor, h_tensor, c_id, plot_list, l_name, mode, r_range, composite, rf, "get_max_reference")
                                   
                if stats:
                    targets, _ = self.compute_stats(c_id, l_name, mode, top_N, mean_N)
                    for t in targets:
                        stat_index = f"{c_id}:{t}"
                        s_tensor, h_tensor = self.get_stats_reference(c_id, l_name, t, mode, r_range, composite, rf, None, batch_size)[stat_index]
                        self._save_precomputed(s_tensor, h_tensor, stat_index, plot_list, l_name, mode, r_range, composite, rf, "get_stats_reference")

                pbar.update(1)

            pbar.close()

       