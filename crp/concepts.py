import torch
import numpy as np
from typing import List


class Concept:

    def mask(self, batch_id, concept_ids, layer_name):

        raise NotImplementedError("<Concept> class must be implemented!")

    def mask_rf(self, neuron_ids, layer_name):

        raise NotImplementedError("<Concept> class must be implemented!")

    def reference_sampling(self, relevance, layer_name: str = None, max_target: str = "sum", abs_norm=True):

        raise NotImplementedError("<Concept> class must be implemented!")

    def get_rf_indices(self, output_shape, layer_name):

        raise NotImplementedError("<Concept> class must be implemented!")

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        raise NotImplementedError("<Concept> class must be implemented!")


class ChannelConcept(Concept):
    """
    Conv and Linear layers
    """

    @staticmethod
    def mask(batch_id: int, concept_ids: List, layer_name=None):
        """
        Modifies gradient (replaced by zennit by attribution) at first dimension with index 'batch_id'

        Parameters:
        ----------
        concept_ids: list of integer values / integer lists corresponding to channel indices. Batch dimension of data should
            match length of list.
        """

        def mask_fct(grad):

            mask = torch.zeros_like(grad[batch_id])
            mask[concept_ids] = 1
            grad[batch_id] = grad[batch_id] * mask

            return grad

        return mask_fct

    @staticmethod
    def mask_rf(batch_id, neuron_ids, layer_name=None):

        def mask_fct(grad):

            grad_shape = grad.shape
            grad = grad.view(*grad_shape[:2], -1)

            mask = torch.zeros_like(grad[batch_id])
            mask[0, neuron_ids] = 1

            grad[batch_id] = grad[batch_id] * mask
            return grad.view(grad_shape)

        return mask_fct

    def get_rf_indices(self, output_shape, layer_name=None):

        if len(output_shape) == 1:
            return [0]
        else:
            end = np.prod(output_shape[1:])
            return np.arange(0, end)

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l

    def reference_sampling(self, relevance, layer_name: str = None, max_target: str = "sum", abs_norm=True):
        """
        Parameters:
            max_target: str. Either 'sum' or 'max'.
        """

        # position of receptive field neuron
        rel_l = relevance.view(*relevance.shape[:2], -1)
        rf_neuron = torch.argmax(rel_l, dim=-1)

        # channel maximization target
        if max_target == "sum":

            rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        elif max_target == "max":
            rel_l = torch.gather(rel_l, 0, rf_neuron)

        else:
            raise ValueError("<max_target> supports only 'max' or 'sum'.")

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)
        
        d_ch_sorted = torch.argsort(rel_l, dim=0, descending=True)
        rel_ch_sorted = torch.gather(rel_l, 0, d_ch_sorted)
        rf_ch_sorted = torch.gather(rf_neuron, 0, d_ch_sorted)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted
