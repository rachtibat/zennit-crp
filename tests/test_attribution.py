
import torch.nn as nn
import torch
from zennit.composites import EpsilonPlus
from crp.helper import get_layer_names
from crp.attribution import CondAttribution


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

class LinearCondAttribution(CondAttribution):

    def attribution_modifier(self, data, on_device=None):

        heatmap = data.grad.detach()
        heatmap = heatmap.to(on_device) if on_device else heatmap
        return heatmap


def test_simple_attribution():

    model = SimpleModel()
    
    attribution = LinearCondAttribution(model)

    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0]}]
    composite = EpsilonPlus()

    layer_names = get_layer_names(model, [nn.Linear])

    attr = attribution(inp, conditions, composite, layer_names)

    assert torch.allclose(attr.heatmap, torch.tensor([-11.0, 21.0]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([1.0, 4.0]))
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([5.0, 0.0]))
