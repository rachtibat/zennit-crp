from zennit.composites import NameMapComposite
from zennit.core import Composite
from crp.hooks import MaskHook
from crp.concepts import Concept, ChannelConcept
from crp.graph import ModelGraph
from typing import Callable, List, Dict, Union
import torch
import warnings
import numpy as np
import math
from tqdm import tqdm
from collections import namedtuple

attrResult = namedtuple("AttributionResults", "heatmap, activations, relevances, prediction")
attrGraphResult = namedtuple("AttributionGraphResults", "nodes, connections")


class CondAttribution:

    def __init__(self, model: torch.nn.Module, device: torch.device = None) -> None:

        self.MODEL_OUTPUT_NAME = "y"

        self.device = next(model.parameters()
                           ).device if device is None else device
        self.model = model

    def backward_initialization(self, prediction, target_list, init_rel, layer_name, retain_graph=False):
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
            layer_name: str 
                either 'y' if backward pass starts at model output, otherwise name of start_layer
            retain_graph: boolean
                used in generator expression to create a persistent computation backward graph
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

        torch.autograd.backward((prediction,), (output_selection.to(prediction),),
                                retain_graph=retain_graph)

    def attribution_modifier(self, data):

        heatmap = data.grad.detach().cpu()
        return torch.sum(heatmap, dim=1)

    def check_arguments(self, data, conditions, start_layer):

        if not data.requires_grad:
            raise ValueError(
                "requires_grad attribute of <data> must be True.")

        for cond in conditions:
            if self.MODEL_OUTPUT_NAME not in cond and start_layer is None:
                raise ValueError(
                    f"Either {self.MODEL_OUTPUT_NAME} in <conditions> or <start_layer> must be defined.")

            if self.MODEL_OUTPUT_NAME in cond and start_layer is not None:
                warnings.warn(
                    f"You defined a condition for {self.MODEL_OUTPUT_NAME} that has no effect, since the <start_layer> {start_layer}"
                    " is provided where the backward pass begins. If this behavior is not wished, remove <start_layer>.")

        data.grad = None
        data.retain_grad()

    def broadcast(self, data, conditions):

        len_data, len_cond = len(data), len(conditions)

        if len_data == len_cond:
            return data, conditions

        if len_cond > 1:
            data = torch.repeat_interleave(data, len_cond, dim=0)
        if len_data > 1:
            conditions = conditions * len_data

        return data, conditions

    def register_mask_fn(self, hook, mask_map, b_index, c_indices, l_name):

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
            on_device: str = None) -> attrResult:

        data, conditions = self.broadcast(data, conditions)

        self.check_arguments(data, conditions, start_layer)

        handles, layer_out = self._append_recording_layer_hooks(
            record_layer, start_layer)

        hook_map, y_targets = {}, []
        for i, cond in enumerate(conditions):
            for l_name, indices in cond.items():
                if l_name == self.MODEL_OUTPUT_NAME:
                    y_targets.append(indices)
                else:
                    if l_name not in hook_map:
                        hook_map[l_name] = MaskHook([])
                    self.register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data)
                pred = layer_out[start_layer]
                self.backward_initialization(pred, None, init_rel, start_layer)

            else:
                pred = modified(data)
                self.backward_initialization(pred, y_targets, init_rel, self.MODEL_OUTPUT_NAME)

            attribution = self.attribution_modifier(data)
            activations, relevances = {}, {}
            if len(layer_out) > 0:
                activations, relevances = self._collect_hook_activation_relevance(
                    layer_out)
            [h.remove() for h in handles]

        return attrResult(attribution, activations, relevances, pred)

    def generate(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = ChannelConcept.mask, start_layer: str = None, init_rel=None,
            batch_size=10, on_device=None, verbose=True) -> attrResult:

        self.check_arguments(data, conditions, start_layer)

        handles, layer_out = self._append_recording_layer_hooks(
            record_layer, start_layer)

        # register on all layers in layer_map an empty hook
        hook_map = {}
        for cond in conditions:
            for l_name in cond.keys():
                if l_name not in hook_map:
                    hook_map[l_name] = MaskHook([])
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

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data_batch)
                pred = layer_out[start_layer]
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
                            self.register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)

                if b == batches-1:
                    # last batch may have len(y_targets) != batch_size. Padded part is ignored later.
                    if not start_layer:
                        y_targets.extend([y_targets[0] for i in range(batch_size-len(y_targets))])
                    batch_size = len(cond_batch)

                layer_name = start_layer if start_layer else self.MODEL_OUTPUT_NAME
                self.backward_initialization(pred, y_targets, init_rel, layer_name, True)

                attribution = self.attribution_modifier(data_batch)
                activations, relevances = {}, {}
                if len(layer_out) > 0:
                    activations, relevances = self._collect_hook_activation_relevance(
                        layer_out, on_device, batch_size)

                yield attrResult(attribution[:batch_size], activations, relevances, pred[:batch_size])

                self._reset_gradients(data_batch)
                [hook.fn_list.clear() for hook in hook_map.values()]

        [h.remove() for h in handles]

        if verbose:
            pbar.close()

    @staticmethod
    def generate_hook(layer_name, layer_out):
        def get_tensor_hook(module, input, output):
            layer_out[layer_name] = output
            output.retain_grad()

        return get_tensor_hook

    def _append_recording_layer_hooks(self, record_l_names: List, start_layer):

        handles = []
        layer_out = {}
        record_l_names = record_l_names.copy()

        if start_layer is not None and start_layer not in record_l_names:
            record_l_names.append(start_layer)

        for name, layer in self.model.named_modules():

            if name == self.MODEL_OUTPUT_NAME:
                raise ValueError(
                    "No layer name should match the constant for the identifier of the model output."
                    "Please change the layer name or the OUTPUT_NAME constant of the object."
                    "Note, that the condition set then references to the output with OUTPUT_NAME and no longer 'y'.")

            if name in record_l_names:

                h = layer.register_forward_hook(self.generate_hook(name, layer_out))
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
            on_device: str or None
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

            if layer_out[name].grad is not None:
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
                    mask_map=self.mask_map, start_layer=start_layer, batch_size=batch_size, verbose=verbose):

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
