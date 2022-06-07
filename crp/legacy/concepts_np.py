import torch
import numpy as np
import numba

class Concept:

    def mask(self, concept_ids, layer_name):

        raise NotImplementedError("<Concept> class must be implemented!")

    def attribute(self, relevance, layer_name: str=None, max_target: str="sum", abs_norm=True):

        raise NotImplementedError("<Concept> class must be implemented!")

    #TODO: here?
    def reference_samples(self, concept_ids):
        """
        return 
        """
        raise NotImplementedError("<Concept> class must be implemented!")
        


class ChannelConcept(Concept):
    """
    Conv and Linear layers
    """

    def mask(self, concept_ids=[], layer_name=None):
        """

        Parameters:
        ----------
        concept_ids: list of integer values / integer lists corresponding to channel indices. Batch dimension of data should
            match length of list.
        """

        def mask_fct(grad):
            if len(concept_ids) != grad.shape[0]:
                raise ValueError("len of <concept_ids> should match batch size.")
            
            mask = torch.zeros_like(grad)
            for i, channels in enumerate(concept_ids):
                mask[i, channels] = 1

            return grad * mask
        
        return mask_fct

    #TODO: attribute act
    def attribute(self, relevance, layer_name: str=None, max_target: str="sum", abs_norm=True):
        """
        Parameters:
            max_target: str. Either 'sum' or 'max'.
        """

        # position of receptive field neuron
        rel_l = relevance.reshape(*relevance.shape[:2], -1)
        rf_neuron = np.argmax(rel_l, axis=-1)

        # channel maximization target
        if max_target == "sum":

            #ch_axes = tuple(np.arange(2, 2+len(relevance.shape[2:])))
            #ch_axes = tuple([s for s in range(2, 2+len(relevance.shape[2:]))])

            rel_l = np.sum(relevance.reshape(*relevance.shape[:2], -1), axis=-1)

        elif max_target == "max":
            rel_l = np.take_along_axis(rel_l, rf_neuron, axis=0)
        
        else:
            raise ValueError("<max_target> supports only 'max' or 'sum'.")

        if abs_norm:
            rel_l = rel_l / (abs(rel_l).sum(-1).reshape(-1, 1) + 1e-10)

        d_ch_sorted = np.flip(np.argsort(rel_l, axis=0), axis=0)
        rel_ch_sorted = np.take_along_axis(rel_l, d_ch_sorted, axis=0)
        rf_ch_sorted = np.take_along_axis(rf_neuron, d_ch_sorted, axis=0)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted

        
    

if __name__ == "__main__":

    from crp.attribution import CondAttribution
    from torchvision.models.vgg import vgg16_bn
    from zennit.types import Convolution
    from zennit.composites import EpsilonPlusFlat
    from zennit.canonizers import SequentialMergeBatchNorm

    from crp.helper import get_layer_names
    from crp.concepts import ChannelConcept

    device = "cuda:0"

    batch_size = 32
    data = torch.randn((batch_size, 3, 224, 224), requires_grad=True).to(device)

    model = vgg16_bn(True).to(device)
    model.eval()

    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])

    attribution = CondAttribution(model)
    c = ChannelConcept()

    conv_layers = get_layer_names(model, [torch.nn.Conv2d])[:5]

    conditions = {"y": c.mask(np.random.randint(0, 10, batch_size))}

    heat, act, rel = attribution(
        data, conditions, composite, conv_layers)

    d_c_sorted, rel_c_sorted, rf_c_sorted = c.reference_sampling(rel[conv_layers[0]], max_target="max", abs_norm=True)

    data_indices = np.arange(100, 100+batch_size)
    np.take(data_indices, d_c_sorted)

