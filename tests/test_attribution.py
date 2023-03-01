
import torch.nn as nn
import torch
from zennit.composites import EpsilonPlus
from crp.helper import get_layer_names
from crp.attribution import CondAttribution
import pytest


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 2, False)
        self.layer2 = nn.Linear(2, 2, False)
        self.layer3 = nn.Linear(4, 1, False)

        self.layer1.weight = nn.Parameter(torch.tensor([[1, 2], [0, 1]], dtype=torch.float32))
        self.layer2.weight = nn.Parameter(torch.tensor([[2, 3], [0, 0]], dtype=torch.float32))
        self.layer3.weight = nn.Parameter(torch.tensor([[1, 4, 5, 0]], dtype=torch.float32))

    def forward(self, x):

        y1 = self.layer1(x)
        y2 = self.layer2(x)

        y3 = torch.cat([y1, y2], dim=1)

        return self.layer3(y3)


class OneDimCondAttribution(CondAttribution):

    def heatmap_modifier(self, data, on_device=None):

        heatmap = data.grad.detach()
        heatmap = heatmap.to(on_device) if on_device else heatmap
        return heatmap


@pytest.fixture
def simple_cond_attribution():
    model = SimpleModel()
    return model, OneDimCondAttribution(model)


def test_simple_attribution(simple_cond_attribution):

    model, attribution = simple_cond_attribution

    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0]}]
    composite = EpsilonPlus()

    layer_names = get_layer_names(model, [nn.Linear])

    attr = attribution(inp, conditions, composite, layer_names)

    assert torch.allclose(attr.heatmap, torch.tensor([-11.0, 21.0]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([1.0, 4.0]))
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([5.0, 0.0]))


def test_parallel_attribution(simple_cond_attribution):

    model, attribution = simple_cond_attribution

    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0], "layer1": [0], "layer2": []}]
    composite = EpsilonPlus()

    layer_names = get_layer_names(model, [nn.Linear])

    attr = attribution(inp, conditions, composite, layer_names, exclude_parallel=False)

    assert torch.allclose(attr.heatmap, torch.tensor([-1.0, 2.0]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([1.0, 4.0]))
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([5.0, 0.0]))

    conditions = [{"y": [0], "layer1": [0]}]

    inp.grad = None
    attr_p = attribution(inp, conditions, composite, layer_names, exclude_parallel=True)

    assert torch.allclose(attr_p.heatmap, torch.tensor([-1.0, 2.0]))
    assert torch.allclose(attr_p.relevances["layer1"], torch.tensor([1.0, 4.0]))
    assert torch.allclose(attr_p.relevances["layer2"], torch.tensor([0.0, 0.0]))


def test_parallel_cond_attribution(simple_cond_attribution):

    model, attribution = simple_cond_attribution

    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0], "layer2": [0], "layer1": [0]}]
    composite = EpsilonPlus()

    layer_names = get_layer_names(model, [nn.Linear])

    attr = attribution(inp, conditions, composite, layer_names, exclude_parallel=False)

    assert torch.allclose(attr.heatmap, torch.tensor([-11.0, 17.0]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([1.0, 4.0]))
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([5.0, 0.0]))

def test_seq_cond_attribution(simple_cond_attribution):

    model, attribution = simple_cond_attribution

    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0], "layer3": [0], "layer1": [0]}]
    composite = EpsilonPlus()

    layer_names = get_layer_names(model, [nn.Linear])

    attr = attribution(inp, conditions, composite, layer_names, exclude_parallel=True)

    assert torch.allclose(attr.heatmap, torch.tensor([-1.0, 2.0]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([1.0, 4.0]))
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([0.0, 0.0]))
