from zennit.composites import NameMapComposite
from zennit.core import Composite
from crp.masking import MaskHook, ChangingMaskFn
from crp.helper import printProgressBar
from typing import Callable, List, Dict
import torch
import warnings
import numpy as np
import math



def generate_hook(name):
    def get_tensor_hook(module, input, output):
        
        output.retain_grad()

    return get_tensor_hook

class CondAttribution:

    def __init__(
            self, model: torch.nn.Module, input_callback: Callable = None, output_callback: Callable = None,
            device: torch.device = None) -> None:

        self.MODEL_OUTPUT_NAME = "y"

        # TODO: delete callbacks
        if input_callback is not None:
            if callable(input_callback):
                self.input_callback = input_callback
            else:
                raise ValueError(
                    "<input_callback> must be a callable function.")
        else:
            self.input_callback = self._default_callback

        if output_callback is not None:
            if callable(output_callback):
                self.output_callback = output_callback
            else:
                raise ValueError(
                    "<output_callback> must be a callable function.")
        else:
            self.output_callback = self._default_callback

        self.device = next(model.parameters()
                           ).device if device is None else device
        self.model = model

    def _default_callback(self, obj):
        return obj

    def backward_initialization(self, prediction, target, data, layer_name, retain_graph=False):

        if callable(target):
            output_selection = target(prediction)
        elif isinstance(target, torch.Tensor):
            output_selection = prediction * target
        elif isinstance(target, np.ndarray) or isinstance(target, list):
            output_selection = torch.zeros_like(prediction)
            for i, t in enumerate(target):
                output_selection[i, t] = prediction[i, t]

        torch.autograd.backward((prediction,), (output_selection.to(
            self.device),), retain_graph=retain_graph)

    def attribution_modifier(self, data):

        heatmap = data.grad.detach().cpu().numpy()
        return np.sum(heatmap, axis=(0, 1))

    def check_arguments(self, data, conditions, start_layer):

        if not data.requires_grad:
            raise ValueError(
                "requires_grad attribute of <data> must be True.")

        if self.MODEL_OUTPUT_NAME not in conditions and start_layer is None:
            raise ValueError(
                f"Either {self.MODEL_OUTPUT_NAME} in <conditions> or <start_layer> must be defined.")

        if self.MODEL_OUTPUT_NAME in conditions and start_layer is not None:
            warnings.warn(
                f"You defined a condition for {self.MODEL_OUTPUT_NAME} that has no effect, since the <start_layer> {start_layer}"
                " is provided where the backward pass begins. If this behavior is not wished, remove <start_layer>.")

        data.retain_grad()

    def __call__(self, data: torch.tensor, conditions: Dict[str, Callable], composite: Composite = None, record_layer: List[str] = [],
                 start_layer: str = None, append_hook: callable=generate_hook):

        self.check_arguments(data, conditions, start_layer)

        handles, layer_out = self.append_recording_layer_hooks(append_hook, record_layer, start_layer)

        name_map = []
        for name, mask_fct in conditions.items():
            if name != self.MODEL_OUTPUT_NAME:
                name_map.append(([name], MaskHook(mask_fct)))

        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data)
                pred = layer_out[start_layer]
                target = conditions[start_layer]
                self.backward_initialization(pred, target, data, start_layer)

            else:
                pred = modified(data)
                target = conditions[self.MODEL_OUTPUT_NAME]
                self.backward_initialization(
                    pred, target, data, self.MODEL_OUTPUT_NAME)

            attribution = self.attribution_modifier(data)
            activations, relevances = {}, {}
            if len(layer_out) > 0:
                activations, relevances = self.collect_hook_activation_relevance(
                    layer_out)

            if len(handles) > 0:
                [h.remove() for h in handles]

        return attribution, activations, relevances

    def generate(
            self, data: torch.tensor, conditions: Dict[str, List],
            mask_super_fn: Callable[[List, str],
                                    Callable],
            composite: Composite = None, record_layer: List[str] = [],
            start_layer: str = None, batch_size=10, verbose=True):

        self.check_arguments(data, conditions, start_layer)

        handles, layer_out = self.append_recording_layer_hooks(
            record_layer, start_layer)

        name_map, hook_register = [], {}

        concept_length = len(next(iter(conditions.values())))
        for name, c_ids in conditions.items():
            if len(c_ids) != concept_length:
                raise IndexError(
                    "length of each list in <conditions> must be equal.")

            if name != self.MODEL_OUTPUT_NAME:
                # ChangingMaskFn's attribute <fn> contains the callback
                # workaround to allow passing arguments by reference
                hook = MaskHook(ChangingMaskFn())
                name_map.append(([name], hook))
                hook_register[name] = hook

        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        if concept_length > batch_size:
            batches = math.ceil(concept_length / batch_size)
        else:
            batches = 1
            batch_size = concept_length

        data_batch = self.expand_data_to_batch(data, batch_size)

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data_batch)
                pred = layer_out[start_layer]
            else:
                pred = modified(data_batch)

            for b in range(batches):

                if verbose:
                    printProgressBar(
                        b, batches, prefix='Progress:', suffix='Complete')

                for name, c_ids in conditions.items():
                    if name != self.MODEL_OUTPUT_NAME:
                        mask_fn = mask_super_fn(
                            c_ids[b * batch_size: (b + 1) * batch_size], name)
                        # ChangingMaskFn's attribute <fn> accessed here
                        hook_register[name].mask_fn.fn = mask_fn

                if start_layer:
                    target_b = conditions[start_layer][b *
                                                       batch_size: (b + 1) * batch_size]
                    target_fn = mask_super_fn(target_b, start_layer)
                    self.backward_initialization(
                        pred, target_fn, data, start_layer, True)
                else:
                    target_b = conditions[self.MODEL_OUTPUT_NAME][b *
                                                                  batch_size: (b + 1) * batch_size]
                    target_fn = mask_super_fn(target_b, self.MODEL_OUTPUT_NAME)
                    self.backward_initialization(
                        pred, target_fn, data, self.MODEL_OUTPUT_NAME, True)

                attribution = self.attribution_modifier(data_batch)
                activations, relevances = {}, {}
                if len(layer_out) > 0:
                    activations, relevances = self.collect_hook_activation_relevance(
                        layer_out)

                yield attribution, activations, relevances

                self.reset_gradients(data)

        if verbose:
            printProgressBar(batches, batches,
                             prefix='Progress:', suffix='Complete')
        [h.remove() for h in handles]

    def expand_data_to_batch(self, data, batch_size):

        repeat_shape = [1] * len(data.shape[1:])
        data_batch = data.repeat(batch_size, *repeat_shape).detach()
        data_batch.requires_grad = True
        return data_batch

    def append_recording_layer_hooks(self, generate_hook, record_l_names: List, start_layer):

        handles = []
        layer_out = {}
        record_l_names = record_l_names.copy()

        if start_layer is not None and start_layer not in record_l_names:
            record_l_names.append(start_layer)

        for name, layer in self.model.named_modules():
            if name in record_l_names:
                h = layer.register_forward_hook(generate_hook(name))
                handles.append(h)
                record_l_names.remove(name)

            if name == self.MODEL_OUTPUT_NAME:
                raise ValueError(
                    "No layer name should match the constant for the identifier of the model output."
                    "Please change the layer name or the OUTPUT_NAME constant of the class."
                    "Note, that the condition set then references to the output with OUTPUT_NAME and no longer 'y'.")

        if start_layer in record_l_names:
            raise KeyError(f"<start_layer> {start_layer} not found in model.")
        if len(record_l_names) > 0:
            warnings.warn(
                f"Some layer names not found in model: {record_l_names}.")

        return handles, layer_out

    def collect_hook_activation_relevance(self, layer_out):

        relevances = {}
        activations = {}
        for name in layer_out:
            activations[name] = layer_out[name].detach().cpu().numpy()  # TODO
            #activations[name].requires_grad = False

            if layer_out[name].grad is not None:
                relevances[name] = layer_out[name].grad.detach().cpu().numpy()
                #relevances[name].requires_grad = False
                layer_out[name].grad = None

        return activations, relevances

    def reset_gradients(self, data):
        """
        custom zero_grad() function
        """

        for p in self.model.parameters():
            p.grad = None

        data.grad = None


if __name__ == "__main__":

    from torchvision.models.vgg import vgg16_bn
    from zennit.types import Convolution
    from zennit.composites import EpsilonPlusFlat
    from zennit.canonizers import SequentialMergeBatchNorm

    from crp.helper import get_layer_names
    from crp.concepts import ChannelConcept

    device = "cuda:0"

    data = torch.randn((1, 3, 224, 224), requires_grad=True).to(device)

    model = vgg16_bn(True).to(device)
    model.eval()

    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])

    attribution = CondAttribution(model)
    c = ChannelConcept()

    conv_layers = get_layer_names(model, [torch.nn.Conv2d])[:2]

    conditions = {"features.27": c.mask([50]), "y": c.mask([1])}

    heat, act, rel = attribution(
        data, conditions, composite, conv_layers, start_layer="features.27")

    conditions = {"features.27": np.arange(0, 512), "y": np.repeat(1, 512)}

    lol = []
    for heat, act, rel in attribution.generate(data, conditions, c.mask, composite, start_layer="features.27"):

        lol.append(rel["features.27"])

    print("hi")
