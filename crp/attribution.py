from zennit.composites import NameMapComposite
from zennit.core import Composite
from crp.hooks import MaskHook
from crp.concepts import Concept, ChannelConcept
from crp.graph import ModelGraph
from typing import Callable, List, Dict, Union, Tuple
import torch
import warnings
import numpy as np
import math
from tqdm import tqdm
from collections import namedtuple

attrResult = namedtuple("AttributionResults", "heatmap, activations, relevances, prediction")
attrGraphResult = namedtuple("AttributionGraphResults", "nodes, connections")


class CondAttribution:

    def __init__(self, model: torch.nn.Module, device: torch.device = None, overwrite_data_grad=True, no_param_grad=True) -> None:
        """
        This class contains the functionality to compute conditional attributions.

        Parameters:
        ----------
        model: torch.nn.Module
        device: torch.device
            specifies where the model and subsequent computation takes place.
        overwrite_data_grad: boolean
            If True, the .grad attribute of the 'data' argument is set to None before each __call__.
        no_param_grad: boolean
            If True, sets the requires_grad attribute of all model parameters to zero, to reduce the GPU memory footprint.
        """

        self.MODEL_OUTPUT_NAME = "y"

        self.device = next(model.parameters()).device if device is None else device
        self.model = model
        self.overwrite_data_grad = overwrite_data_grad

        if no_param_grad:
            self.model.requires_grad_(False)


    def backward(self, pred, grad_mask, partial_backward, layer_names, layer_out, generate=False):

        if partial_backward and len(layer_names) > 0:

            wrt_tensor, grad_tensors = pred, grad_mask.to(pred)

            for l_name in layer_names:

                inputs = layer_out[l_name]

                try:
                    grad = torch.autograd.grad(wrt_tensor, inputs=inputs, grad_outputs=grad_tensors, retain_graph=True)
                except RuntimeError as e:
                    if "allow_unused=True" not in str(e):
                        raise e
                    else:
                        raise RuntimeError(
                            "The layer names must be ordered according to their succession in the model if 'exclude_parallel'=True."
                            " Please make sure to start with the last and end with the first layer in each condition dict. In addition,"
                            " parallel layers can not be used in one condition.")

                # TODO: necessary?
                if grad is None:
                    raise RuntimeError(
                        "The layer names must be ordered according to their succession in the model if 'exclude_parallel'=True."
                        " Please make sure to start with the last and end with the first layer in each condition dict. In addition,"
                        " parallel layers can not be used in one condition.")

                wrt_tensor, grad_tensors = layer_out[l_name], grad

            torch.autograd.backward(wrt_tensor, grad_tensors, retain_graph=generate)

        else:

            torch.autograd.backward(pred, grad_mask.to(pred), retain_graph=generate)

    def relevance_init(self, prediction, target_list, init_rel):
        """

        Parameters:
        -----------
            prediction: torch.Tensor
                output of model forward pass
            target_list: list/numpy.ndarray or None
                list of all 'y' values of condition dictionaries. Indices are used to set the
                initial relevance to prediction values. If target_list is None and init_rel is None,
                relevance is initialized at all indices with prediction values. If start_layer is
                used, target_list is set to None.
            init_rel: torch.Tensor or None
                used to initialize relevance instead of prediction. If None, target_list is used.
                Please make sure to choose the right shape.
        """

        if callable(init_rel):
            output_selection = init_rel(prediction)
        elif isinstance(init_rel, torch.Tensor):
            output_selection = init_rel
        elif isinstance(init_rel, (int, np.integer)):
            output_selection = torch.full(prediction.shape, init_rel)
        else:
            output_selection = prediction

        if target_list:
            mask = torch.zeros_like(output_selection)
            for i, targets in enumerate(target_list):
                mask[i, targets] = output_selection[i, targets]
            output_selection = mask

        return output_selection

    def heatmap_modifier(self, data, on_device=None):

        heatmap = data.grad.detach()
        heatmap = heatmap.to(on_device) if on_device else heatmap
        return torch.sum(heatmap, dim=1)

    def broadcast(self, data, conditions) -> Tuple[torch.Tensor, Dict]:

        len_data, len_cond = len(data), len(conditions)

        if len_data == len_cond:
            data.retain_grad()
            return data, conditions

        if len_cond > 1:
            data = torch.repeat_interleave(data, len_cond, dim=0)
        if len_data > 1:
            conditions = conditions * len_data

        data.retain_grad()
        return data, conditions

    def _check_arguments(self, data, conditions, start_layer, exclude_parallel, init_rel):

        if not data.requires_grad:
            raise ValueError(
                "requires_grad attribute of 'data' must be True.")

        if self.overwrite_data_grad:
            data.grad = None
        elif data.grad is not None:
            warnings.warn("'data' already has a filled .grad attribute. Set to None if not intended or set 'overwrite_grad' to True.")

        distinct_cond = set()
        for cond in conditions:
            if self.MODEL_OUTPUT_NAME not in cond and start_layer is None and init_rel is None:
                raise ValueError(
                    f"Either {self.MODEL_OUTPUT_NAME} in 'conditions' or 'start_layer' or 'init_rel' must be defined.")

            if self.MODEL_OUTPUT_NAME in cond and start_layer is not None:
                warnings.warn(
                    f"You defined a condition for {self.MODEL_OUTPUT_NAME} that has no effect, since the 'start_layer' {start_layer}"
                    " is provided where the backward pass begins. If this behavior is not wished, remove 'start_layer'.")

            if exclude_parallel:

                if len(distinct_cond) == 0:
                    distinct_cond.update(cond.keys())
                elif distinct_cond ^ set(cond.keys()):
                    raise ValueError("If the 'exclude_parallel' flag is set to True, each condition dict must contain the"
                                     " same layer names. (This limitation does not apply to the __call__ method)")


    def _register_mask_fn(self, hook, mask_map, b_index, c_indices, l_name):

        if callable(mask_map):
            mask_fn = mask_map(b_index, c_indices, l_name)
        elif isinstance(mask_map, Dict):
            mask_fn = mask_map[l_name](b_index, c_indices, l_name)
        else:
            raise ValueError("<mask_map> must be a dictionary or callable function.")

        hook.fn_list.append(mask_fn)


    def __call__(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = ChannelConcept.mask, start_layer: str = None, init_rel=None,
            on_device: str = None, exclude_parallel=True) -> attrResult:

        """
        Computes conditional attributions by masking the gradient flow of PyTorch (that is replaced by zennit with relevance values).
        The relevance distribution rules (as for LRP e.g.) are described in the zennit 'composite'. Relevance can be initialized at
        the model output or 'start_layer' with the 'init_rel' argument.
        How the relevances are masked is determined by the 'conditions' as well as the 'mask_map'. In addition, 'exclude_parallel'=True,
        restricts the PyTorch gradient flow so that it does not enter into parallel layers (shortcut connections) of the layers mentioned
        in the 'conditions' dictionary.
        The name of the model output is designated with self.MODEL_OUTPUT_NAME ('y' per default) and can be used inside 'conditions'.

        Parameters:
        -----------

        data: torch.Tensor
            Input sample for which a conditional heatmap is computed
        conditions: list of dict
            The key of a dict are string layer names and their value is a list of integers describing the concept (channel, neuron) index.
            In general, the values are passed to the 'mask_map' function as 'concept_ids' argument.
        composite: zennit Composite
            Object that describes how relevance is distributed. Should contain a suitable zennit Canonizer.
        mask_map: dict of callable or callable
            The keys of the dict are string layer names and the values functions that implement gradient masking. If no dict is used,
            all layers are masked according to the same function. 
            The 'conditions' values are passed into the function as 'concept_ids' argument.
        start_layer: (optional) str
            Layer name where to start the backward pass instead of starting at the model output. 
            If set, 'init_rel' modifies the tensor at 'start_layer' instead and a condition containing self.MODEL_OUTPUT_NAME is ignored.
        init_rel: (optional) torch.Tensor, int or callable
            Initializes the relevance distribution process as described in the LRP algorithm e.g. The callable must have the signature
            callable(activations).
            Per default, relevance is initialized with the logit activation before a non-linearity.
        on_device: (optional) str
            On which device (cpu, cuda) to save the heatmap, intermediate activations and relevances.
            Per default, everything is kept on the same device as the model parameters.
        exclude_parallel: boolean
            If set, the PyTorch gradient flow is restricted so that it does not enter into parallel layers (shortcut connections) 
            of the layers mentioned in the 'conditions' dictionary. Useful to get the sole contribution of a specific concept.

        Returns:
        --------
         
        attrResult: namedtuple object
            Contains the attributes 'heatmap', 'activations', 'relevances' and 'prediction'.
            'heatmap': torch.Tensor
                Output of the self.attribution_modifier method that defines how 'data'.grad is processed.
            'activations': dict of str and torch.Tensor
                The keys are the layer names and values are the activations
            'relevances': dict of str and torch.Tensor
                The keys are the layer names and values are the relevances
            'prediction': torch.Tensor
                The model prediction output. If 'start_layer' is set, 'prediction' is the layer activation.       
        """
        
        if exclude_parallel:
            return self._conditions_wrapper(data, conditions, composite, record_layer, mask_map, start_layer, init_rel, on_device, True)
        else:
            return self._attribute(data, conditions, composite, record_layer, mask_map, start_layer, init_rel, on_device, False)

    def _conditions_wrapper(self, *args):
        """
        Since 'exclude_parallel'=True requires that the condition set contains only the same layer names,
        the list is divided into distinct lists that all contain the same layer name.
        """

        data, conditions = args[:2]

        relevances, activations = {}, {}
        heatmap, prediction = None, None

        dist_conds = self._separate_conditions(conditions)

        for dist_layer in dist_conds:

            attr = self._attribute(data, dist_conds[dist_layer], *args[2:])

            for l_name in attr.relevances:
                if l_name not in relevances:
                    relevances[l_name] = attr.relevances[l_name]
                    activations[l_name] = attr.activations[l_name]
                else:
                    relevances[l_name] = torch.cat([relevances[l_name], attr.relevances[l_name]], dim=0)
                    activations[l_name] = torch.cat([activations[l_name], attr.activations[l_name]], dim=0)

            if heatmap is None:
                heatmap = attr.heatmap
                prediction = attr.prediction
            else:
                heatmap = torch.cat([heatmap, attr.heatmap], dim=0)
                prediction = torch.cat([prediction, attr.prediction], dim=0)

        return attrResult(heatmap, activations, relevances, prediction)

    def _separate_conditions(self, conditions):
        """
        Finds identical subsets of layer names inside 'conditions'
        """

        distinct_cond = dict()
        for cond in conditions:
            cond_set = frozenset(cond.keys())

            if cond_set in distinct_cond:
                distinct_cond[cond_set].append(cond)
            else:
                distinct_cond[cond_set] = [cond]

        return distinct_cond


    def _attribute(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = ChannelConcept.mask, start_layer: str = None, init_rel=None,
            on_device: str = None, exclude_parallel=True) -> attrResult:
        """
        Computes the actual attributions as described in __call__ method docstring.
        exclude_parallel: boolean
            If set, all layer names in 'conditions' must be identical. This limitation does not apply to the __call__ method.
        """
        data, conditions = self.broadcast(data, conditions)

        self._check_arguments(data, conditions, start_layer, exclude_parallel, init_rel)

        hook_map, y_targets, cond_l_names = {}, [], []
        for i, cond in enumerate(conditions):
            for l_name, indices in cond.items():
                if l_name == self.MODEL_OUTPUT_NAME:
                    y_targets.append(indices)
                else:
                    if l_name not in hook_map:
                        hook_map[l_name] = MaskHook([])
                    self._register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)
                    if l_name not in cond_l_names:
                        cond_l_names.append(l_name)

        handles, layer_out = self._append_recording_layer_hooks(record_layer, start_layer, cond_l_names)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data)
                pred = layer_out[start_layer]
                grad_mask = self.relevance_init(pred.detach().clone(), None, init_rel)
                if start_layer in cond_l_names:
                    cond_l_names.remove(start_layer)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)

            else:
                pred = modified(data)
                grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)

            attribution = self.heatmap_modifier(data, on_device)
            activations, relevances = {}, {}
            if len(layer_out) > 0:
                activations, relevances = self._collect_hook_activation_relevance(layer_out, on_device)
            [h.remove() for h in handles]

        return attrResult(attribution, activations, relevances, pred)

    def generate(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = ChannelConcept.mask, start_layer: str = None, init_rel=None,
            batch_size=10, on_device=None, exclude_parallel=True, verbose=True) -> attrResult:
        """
        Computes several conditional attributions for single data point by broadcasting 'data' to length 'batch_size' and
        iterating through the 'conditions' list with stepsize 'batch_size'. The model forward pass is performed only once and 
        the backward graph kept in memory in order to double the performance.
        Please refer to the docstring of the __call__ method.

        batch_size: int
            batch size of each forward and backward pass
        exclude_parallel: boolean
            If set, all layer names in 'conditions' must be identical. This limitation does not apply to the __call__ method.
        verbose: boolean
            If set, a progressbar is displayed.
        """

        self._check_arguments(data, conditions, start_layer, exclude_parallel, init_rel)

        # register on all layers in layer_map an empty hook
        hook_map, cond_l_names = {}, []
        for cond in conditions:
            for l_name in cond.keys():
                if l_name not in hook_map:
                    hook_map[l_name] = MaskHook([])
                if l_name != self.MODEL_OUTPUT_NAME and l_name not in cond_l_names:
                    cond_l_names.append(l_name)

        handles, layer_out = self._append_recording_layer_hooks(record_layer, start_layer, cond_l_names)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        cond_length = len(conditions)
        if cond_length > batch_size:
            batches = math.ceil(cond_length / batch_size)
        else:
            batches = 1
            batch_size = cond_length

        data_batch = torch.repeat_interleave(data, batch_size, dim=0)
        data_batch.grad = None
        data_batch.retain_grad()
        retain_graph = True

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data_batch)
                pred = layer_out[start_layer]
                if start_layer in cond_l_names:
                    cond_l_names.remove(start_layer)

            else:
                pred = modified(data_batch)

            if verbose:
                pbar = tqdm(total=batches, dynamic_ncols=True)

            for b in range(batches):

                if verbose:
                    pbar.update(1)

                cond_batch = conditions[b * batch_size: (b + 1) * batch_size]

                y_targets = []
                for i, cond in enumerate(cond_batch):
                    for l_name, indices in cond.items():
                        if l_name == self.MODEL_OUTPUT_NAME:
                            y_targets.append(indices)
                        else:
                            self._register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)

                if b == batches-1:
                    # last batch may have len(y_targets) != batch_size. Padded part is ignored later.
                    # and backward graph is freed with retain_graph=False
                    if not start_layer:
                        y_targets.extend([y_targets[0] for i in range(batch_size-len(y_targets))])
                    batch_size = len(cond_batch)
                    retain_graph = False

                grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out, retain_graph)

                heatmap = self.heatmap_modifier(data_batch)
                activations, relevances = {}, {}
                if len(layer_out) > 0:
                    activations, relevances = self._collect_hook_activation_relevance(
                        layer_out, on_device, batch_size)

                yield attrResult(heatmap[:batch_size], activations, relevances, pred[:batch_size])

                self._reset_gradients(data_batch)
                [hook.fn_list.clear() for hook in hook_map.values()]

        [h.remove() for h in handles]

        if verbose:
            pbar.close()

    @staticmethod
    def _generate_hook(layer_name, layer_out):
        def get_tensor_hook(module, input, output):
            layer_out[layer_name] = output
            output.retain_grad()

        return get_tensor_hook

    def _append_recording_layer_hooks(self, record_l_names: list, start_layer, cond_l_names):
        """
        applies a forward hook to all layers in record_l_names, start_layer and cond_l_names to record 
        the activations and relevances
        """

        handles = []
        layer_out = {}
        record_l_names = record_l_names.copy()

        for l_name in cond_l_names:
            if l_name not in record_l_names:
                record_l_names.append(l_name)

        if start_layer is not None and start_layer not in record_l_names:
            record_l_names.append(start_layer)

        for name, layer in self.model.named_modules():

            if name == self.MODEL_OUTPUT_NAME:
                raise ValueError(
                    "No layer name should match the constant for the identifier of the model output."
                    "Please change the layer name or the OUTPUT_NAME constant of the object."
                    "Note, that the condition set then references to the output with OUTPUT_NAME and no longer 'y'.")

            if name in record_l_names:
                h = layer.register_forward_hook(self._generate_hook(name, layer_out))
                handles.append(h)
                record_l_names.remove(name)

        if start_layer in record_l_names:
            raise KeyError(f"<start_layer> {start_layer} not found in model.")
        if len(record_l_names) > 0:
            warnings.warn(
                f"Some layer names not found in model: {record_l_names}.")

        return handles, layer_out

    def _collect_hook_activation_relevance(self, layer_out, on_device=None, length=None):
        """

        Parameters:
        ----------
            layer_out: dict
                contains the intermediate layer outputs
            on_device: str
                copy layer_out on cpu or cuda device
            length: int
                copy only first length elements of layer_out. Used for uneven batch sizes.
        """

        relevances = {}
        activations = {}
        for name in layer_out:
            act = layer_out[name].detach()[:length]
            activations[name] = act.to(on_device) if on_device else act
            activations[name].requires_grad = False

            if layer_out[name].grad is None:
                rel = torch.zeros_like(activations[name], requires_grad=False)[:length]
                relevances[name] = rel.to(on_device) if on_device else rel
            else:
                rel = layer_out[name].grad.detach()[:length]
                relevances[name] = rel.to(on_device) if on_device else rel
                relevances[name].requires_grad = False
                layer_out[name].grad = None

        return activations, relevances

    def _reset_gradients(self, data):
        """
        custom zero_grad() function
        """

        for p in self.model.parameters():
            p.grad = None

        data.grad = None


class AttributionGraph:

    def __init__(self, attribution: CondAttribution, graph: ModelGraph, layer_map: Dict[str, Concept]):

        self.attribution = attribution
        self.graph = graph

        self.set_layer_map(layer_map)

    def set_layer_map(self, layer_map):
        """
        set layer map of attribution graph
        """

        self.layer_map = layer_map
        self.mask_map = {l_name: c.mask for l_name, c in layer_map.items()}

    def __call__(
            self, sample, composite, concept_id: int, layer_name, target=None, width: List[int] = [4, 2],
            parent_c_id: int = None, parent_layer: str = None, abs_norm=True, batch_size=16, verbose=True):
        """
        Decomposes a higher-level concept into its lower-level concepts taking advantage of the
        relevance flow of a specific prediction.

        Parameters:
        -----------
        sample: torch.Tensor
        composite: zennit.composites.Composite
        concept_id: int
            index of higher-level concept that is decomposed
        layer_name: str
            name of layer where the higher-level concept is located
        target: None or int
            if defined, decomposes the higher-level concept w.r.t. target prediction
        width: list of integers
            describes how many lower-level concepts per layer are returned. The length
            of the list specifies the number of lower-level layers that are successively decomposed
            following the higher level layer `layer_name`.
        parent_c_id: int
            if the higher-level concept `concept_id` is decomposed in context of another higher concept,
            then this parameter denotes the original higher-level concept.
        parent_layer: str
            layer name of concept with index `parent_c_id`
        abs_norm: boolean
            if True, normalizes the relevance by dividing with the sum of absolute value
        batch_size: int
            maximal batch size

        Returns:
        --------

        nodes: list of tuples
            All concept indices with their layer names present in the attribution graph.
            The first element is the layer name and the second the index.
        connections: dict, keys are str and values are tuples with a length of three
            Describes the connection between two nodes in the graph. 
            The key is the source and the value the target. The first element is the layer name,
            the second the index and the third the relevance value.

        """
        nodes = [(layer_name, concept_id)]
        connections = {}

        if target is not None:
            start_layer = None
        elif parent_layer:
            start_layer = parent_layer
        else:
            start_layer = layer_name

        parent_cond = {}
        if parent_c_id is not None and parent_layer:
            parent_cond[parent_layer] = [parent_c_id]
        else:
            parent_cond[layer_name] = [concept_id]

        if target is not None:
            parent_cond[self.attribution.MODEL_OUTPUT_NAME] = [target]

        cond_tuples = [(layer_name, concept_id)]

        for w in width:

            conditions, input_layers = [], []
            for l_name, c_id in cond_tuples:

                cond = {l_name: [c_id]}
                cond.update(parent_cond)
                conditions.append(cond)

                in_layers = self.graph.find_input_layers(l_name)
                for name in in_layers:
                    if name not in input_layers:
                        input_layers.append(name)

            b, next_cond_tuples = 0, []
            for attr in self.attribution.generate(
                    sample, conditions, composite, record_layer=input_layers,
                    mask_map=self.mask_map, start_layer=start_layer, batch_size=batch_size, verbose=verbose, exclude_parallel=False):

                self._attribute_lower_level(
                    cond_tuples[b * batch_size: (b + 1) * batch_size],
                    attr.relevances, w, nodes, connections, next_cond_tuples, abs_norm)

                b += 1

            cond_tuples = next_cond_tuples

        return attrGraphResult(nodes, connections)

    def _attribute_lower_level(self, cond_tuples, relevances, w, nodes, connections, next_cond_tuples, abs_norm):

        for i, (l_name, c_id) in enumerate(cond_tuples):

            input_layers = self.graph.find_input_layers(l_name)

            for inp_l in input_layers:

                rel = relevances[inp_l][[i]]
                rel_c = self.layer_map[inp_l].attribute(rel, abs_norm=abs_norm)[0]

                c_ids = torch.argsort(rel_c, descending=True)[:w].tolist()
                nodes.extend([(inp_l, id) for id in c_ids])

                next_cond_tuples.extend([(inp_l, id) for id in c_ids])

                if (l_name, c_id) not in connections:
                    connections[(l_name, c_id)] = []
                connections[(l_name, c_id)].extend([(inp_l, id, rel_c[id].item()) for id in c_ids])

        return None
