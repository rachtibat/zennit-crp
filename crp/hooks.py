import weakref
import functools
import torch

from zennit.core import RemovableHandle, RemovableHandleList


class MaskHook:
    '''Mask hooks for adaptive gradient masking or simple modification.'''

    def __init__(self, fn_list):

        self.fn_list = fn_list

    def post_forward(self, module, input, output):
        '''Register a backward-hook to the resulting tensor right after the forward.'''
        hook_ref = weakref.ref(self)

        @functools.wraps(self.backward)
        def wrapper(grad):
            return hook_ref().backward(module, grad)

        if not isinstance(output, tuple):
            output = (output,)

        if output[0].grad_fn is not None:
            # only if gradient required
            output[0].register_hook(wrapper)
        return output[0] if len(output) == 1 else output

    def backward(self, module, grad):
        '''Hook applied during backward-pass'''
        for mask_fn in self.fn_list:
            grad = mask_fn(grad)

        return grad

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        Copies retain the same fn_list list.
        '''
        return self.__class__(fn_list=self.fn_list)

    def remove(self):
        '''When removing hooks, remove all stored mask_fn.'''
        self.fn_list.clear()

    def register(self, module):
        '''Register this instance by registering the neccessary forward hook to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_hook(self.post_forward),
        ])


class FeatVisHook:
    '''Feature Visualization hooks for reference sampling inside forward and backward passes.'''

    def __init__(self, FV, concept, layer_name, dict_inputs, on_device):
        """
        Parameters:
            dict_inputs: contains sample_indices and targets inputs to FV.analyze_activation and FV.analyze_relevance 
        """

        self.FV = FV
        self.concept = concept
        self.layer_name = layer_name
        self.dict_inputs = dict_inputs
        self.on_device = on_device

    def post_forward(self, module, input, output):
        '''Register a backward-hook to the resulting tensor right after the forward.'''

        s_indices, targets = self.dict_inputs["sample_indices"], self.dict_inputs["targets"]
        activation = output.detach().to(self.on_device) if self.on_device else output.detach()
        self.FV.analyze_activation(activation, self.layer_name, self.concept, s_indices, targets)

        hook_ref = weakref.ref(self)

        @functools.wraps(self.backward)
        def wrapper(grad):
            return hook_ref().backward(module, grad)

        if not isinstance(output, tuple):
            output = (output,)

        if output[0].grad_fn is not None:
            # only if gradient required
            output[0].register_hook(wrapper)
        return output[0] if len(output) == 1 else output

    def backward(self, module, grad):
        '''Hook applied during backward-pass'''

        s_indices, targets = self.dict_inputs["sample_indices"], self.dict_inputs["targets"]
        relevance = grad.detach().to(self.on_device) if self.on_device else grad.detach()
        self.FV.analyze_relevance(relevance, self.layer_name, self.concept, s_indices, targets)

        return grad

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        Copies retain the same stored_grads list.
        '''
        return self.__class__(self.FV, self.concept, self.layer_name, self.dict_inputs, self.on_device)

    def remove(self):
        pass

    def register(self, module):
        '''Register this instance by registering the neccessary forward hook to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_hook(self.post_forward),
        ])
