from zennit.composites import NameMapComposite
from zennit.core import Composite
from crp.hooks import MaskHook, ChangingMaskFn
from crp.concepts import Concept
from typing import Callable, List, Dict
import torch
import warnings
import numpy as np
import math
from tqdm import tqdm
from collections import namedtuple

attrResult = namedtuple("AttributionResults", "heatmap, activations, relevances, prediction")

#9.564642901420592 s mean

class CondAttribution:

    def __init__(self, model: torch.nn.Module, layer_map: Dict[str, Concept], device: torch.device = None) -> None:

        self.MODEL_OUTPUT_NAME = "y"

        self.device = next(model.parameters()
                           ).device if device is None else device
        self.model = model
        self.layer_map = layer_map

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
        if init_rel:
            output_selection = init_rel
        elif target_list:
            output_selection = torch.zeros_like(prediction)
            for i, targets in enumerate(target_list):
                output_selection[i, targets] = prediction[i, targets]
        else:
            output_selection = prediction

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

        if len_cond > 1:
            data = torch.repeat_interleave(data, len_cond, dim=0)
        if len_data > 1:
            conditions = conditions * len_data

        return data, conditions

    def __call__(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            start_layer: str = None, init_rel=None, on_device: str = None) -> attrResult:

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
                    mask_fn = self.layer_map[l_name].mask(i, indices, l_name)
                    hook_map[l_name].fn_list.append(mask_fn)

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

    def generate(self, data: torch.tensor, conditions: List[Dict[str, List]], composite: Composite = None, record_layer: List[str] = [],
                 start_layer: str = None, init_rel=None, batch_size=10, on_device=None, verbose=True) -> attrResult:

        self.check_arguments(data, conditions, start_layer)

        handles, layer_out = self._append_recording_layer_hooks(
            record_layer, start_layer)

        # register on all layers in layer_map an empty hook
        hook_map = {l_name: MaskHook([]) for l_name in self.layer_map.keys()}
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
        data_batch.grad = None; data_batch.retain_grad()

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data_batch)
                pred = layer_out[start_layer]
            else:
                pred = modified(data_batch)

            if verbose:
                pbar = tqdm(total=batches, dynamic_ncols=True)

            for b in range(batches-1):

                if verbose:
                    pbar.update(1)

                cond_batch = conditions[b * batch_size: (b + 1) * batch_size]

                y_targets = []
                for i, cond in enumerate(cond_batch):
                    for l_name, indices in cond.items():
                        if l_name == self.MODEL_OUTPUT_NAME:
                            y_targets.append(indices)
                        else:
                            mask_fn = self.layer_map[l_name].mask(i, indices, l_name)
                            hook_map[l_name].fn_list.append(mask_fn)

                if start_layer:
                    self.backward_initialization(
                        pred, y_targets, init_rel, start_layer, True)
                else:
                    self.backward_initialization(
                        pred, y_targets, init_rel, self.MODEL_OUTPUT_NAME, True)

                attribution = self.attribution_modifier(data_batch)
                activations, relevances = {}, {}
                if len(layer_out) > 0:
                    activations, relevances = self._collect_hook_activation_relevance(
                        layer_out, on_device)

                yield attrResult(attribution, activations, relevances, pred)

                self._reset_gradients(data_batch)
                [hook.fn_list.clear() for hook in hook_map.values()]

        [h.remove() for h in handles]

        # last batch can have predictions.shape[0] != batch size
        if verbose:
            pbar.update(1)

        b = batches-1
        last_cond = conditions[b * batch_size: (b + 1) * batch_size]

        yield self.__call__(data, last_cond, composite, record_layer, start_layer, init_rel, on_device)

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
                if name not in self.layer_map:
                    raise ValueError(
                        f"You can only record layers that you defined in the layer_map! {name} is no key in "
                        f"{list(self.layer_map.keys())}. Please change the layer_map attribute of the object.")

                h = layer.register_forward_hook(self.generate_hook(name, layer_out))
                handles.append(h)
                record_l_names.remove(name)

        if start_layer in record_l_names:
            raise KeyError(f"<start_layer> {start_layer} not found in model.")
        if len(record_l_names) > 0:
            warnings.warn(
                f"Some layer names not found in model: {record_l_names}.")

        return handles, layer_out

    def _collect_hook_activation_relevance(self, layer_out, on_device=None):

        relevances = {}
        activations = {}
        for name in layer_out:
            if on_device:
                activations[name] = layer_out[name].detach().to(on_device)
            else:
                activations[name] = layer_out[name].detach()
            activations[name].requires_grad = False

            if layer_out[name].grad is not None:
                if on_device:
                    relevances[name] = layer_out[name].grad.detach().to(on_device)
                else:
                    relevances[name] = layer_out[name].grad.detach()
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
