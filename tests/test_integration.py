import torch.nn as nn
import torch
from zennit.layer import Sum
import torchvision.transforms as T
import numpy as np
from pathlib import Path
from torchvision.datasets import FashionMNIST

from zennit.composites import EpsilonPlus
from zennit.torchvision import SequentialMergeBatchNorm
from crp.visualization import FeatureVisualization
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
import pytest


class FashionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.parallel = nn.Sequential(nn.Conv2d(1, 16, 5, stride=2, bias=False), nn.BatchNorm2d(16))
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()
        self.sum = Sum()
        self.maxpooling = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.parallel is not None:
            identity = self.parallel(identity)

        out = torch.stack([identity, out], dim=-1)
        out = self.sum(out)
        out = self.relu(out)
        out = self.maxpooling(out)
        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu(out)

        return self.linear2(out)


class SplitFashionMNIST(FashionMNIST):

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, False, transform, target_transform, False)

    def _check_exists(self):
        return True

    def _load_data(self):
        data = np.load(Path(self.root, 'fashion_val.npz'))
        return torch.from_numpy(data['val_set']), torch.from_numpy(data['targets'])


@pytest.fixture
def get_fashion_model_data():

    model = FashionModel()
    model.load_state_dict(torch.load("tests/data/fashion_acc=89.ckpt")["state_dict"])
    model.eval()

    val_set = SplitFashionMNIST(
        root='tests/data',
        transform=T.Compose([
            T.ToTensor()
        ])
    )
    return model, val_set


def test_fashion_attribution(get_fashion_model_data):

    model, dataset = get_fashion_model_data

    attribution = CondAttribution(model)

    composite = EpsilonPlus([SequentialMergeBatchNorm()])

    test_sample, target = dataset[0]
    test_sample = test_sample.unsqueeze(0).requires_grad_()

    # test abs() function, broadcasting, parallel and partial heatmaps at once
    conditions = [
        {"y": target},
        {"y": target, "conv2": []},
        {"y": target, "conv2": [2]},
        {"y": target, "parallel.0": []},
        {"y": target, "parallel.0": [], "conv2": [3, 2, 1]}
    ]
    attr = attribution(test_sample, conditions, composite, record_layer=["conv1", "conv2"], init_rel=abs, exclude_parallel=False)

    # -----------------------------------------------
    # generation of test files for documentation only:
    # np.savez_compressed("tests/data/heatmaps_new", heatmaps=attr.heatmap.numpy())
    # np.savez_compressed("tests/data/conv1_relevances_new", conv1_relevances=attr.relevances["conv1"])
    # -----------------------------------------------

    heatmaps = np.load("tests/data/heatmaps.npz")["heatmaps"]
    conv1_relevances = np.load("tests/data/conv1_relevances.npz")["conv1_relevances"]

    assert np.allclose(heatmaps, attr.heatmap.numpy())
    assert np.allclose(conv1_relevances, attr.relevances["conv1"].numpy())

    ### ----------------------- exclude parallel ---------------------------

    test_sample.grad = None
    conditions = [
        {"y": target, "conv2": [3, 2, 1]}
    ]
    attr_p = attribution(test_sample, conditions, composite, record_layer=["conv1", "conv2"], init_rel=abs, exclude_parallel=True)
    
    assert np.allclose(heatmaps[-1], attr_p.heatmap.numpy()[-1])
    assert np.allclose(conv1_relevances[-1], attr_p.relevances["conv1"].numpy()[-1])

def test_fashion_generator_attribution(get_fashion_model_data):

    model, dataset = get_fashion_model_data

    attribution = CondAttribution(model)

    composite = EpsilonPlus([SequentialMergeBatchNorm()])

    test_sample, target = dataset[0]
    test_sample = test_sample.unsqueeze(0).requires_grad_()

    conditions = [
        {"y": target, "parallel.0": [], "conv2": [i]} for i in np.arange(0, 16)
    ]
    heatmaps, relevances = [], []
    for attr in attribution.generate(test_sample, conditions, composite, record_layer=["conv1"], init_rel=abs, batch_size=5, verbose=True, exclude_parallel=False):

        heatmaps.append(attr.heatmap)
        relevances.append(attr.relevances["conv1"])

    heatmaps = torch.cat(heatmaps, dim=0)
    relevances = torch.cat(relevances, dim=0)

    gen_heatmaps = np.load("tests/data/gen_heatmaps.npz")["heatmaps"]
    gen_conv1_relevances = np.load("tests/data/gen_conv1_relevances.npz")["conv1_relevances"]

    assert np.allclose(gen_heatmaps, heatmaps.numpy())
    assert np.allclose(gen_conv1_relevances, relevances.numpy())

    ### ----------------------- exclude parallel ---------------------------

    conditions = [
        {"y": target, "conv2": [i]} for i in np.arange(0, 16)
    ]
    test_sample.grad = None
    heatmaps, relevances = [], []
    for attr in attribution.generate(test_sample, conditions, composite, record_layer=["conv1"], init_rel=abs, batch_size=5, verbose=True, exclude_parallel=True):

        heatmaps.append(attr.heatmap)
        relevances.append(attr.relevances["conv1"])

    heatmaps = torch.cat(heatmaps, dim=0)
    relevances = torch.cat(relevances, dim=0)

    assert np.allclose(gen_heatmaps, heatmaps.numpy())
    assert np.allclose(gen_conv1_relevances, relevances.numpy())
    


def test_fashion_feature_visualization(get_fashion_model_data):

    model, dataset = get_fashion_model_data

    attribution = CondAttribution(model)

    cc = ChannelConcept()
    layer_map = {"conv2": cc, "parallel.0": cc}
    fv = FeatureVisualization(attribution, dataset, layer_map, path="tests/results/fashion_fv")

    composite = EpsilonPlus([SequentialMergeBatchNorm()])
    fv.run(composite, 0, len(dataset))

    ref_c = fv.get_max_reference([0, 2, 10, 11], "conv2", "relevance", (0, 4), composite, rf=True, plot_fn=None)

    # -----------------------------------------------
    # generation of test files for documentation only:
    # save_fv = {}
    # for c_id in ref_c:

    #     imgs = ref_c[c_id][0].numpy()
    #     heats = ref_c[c_id][1].numpy()
    #     save_fv[str(c_id)] = {"imgs": imgs, "heats": heats}

    # np.savez_compressed("tests/data/fv_conv2_rel_new", **save_fv)
    # -----------------------------------------------

    compare = np.load("tests/data/fv_conv2_rel.npz", allow_pickle=True)
    for c_id in ref_c:
        assert np.allclose(compare[str(c_id)].item()["imgs"], ref_c[c_id][0].numpy())
        assert np.allclose(compare[str(c_id)].item()["heats"], ref_c[c_id][1].numpy())

    ref_c = fv.get_max_reference([0, 2, 10, 11], "conv2", "activation", (0, 4), composite, rf=True, plot_fn=None)

    compare = np.load("tests/data/fv_conv2_act.npz", allow_pickle=True)
    for c_id in ref_c:
        assert np.allclose(compare[str(c_id)].item()["imgs"], ref_c[c_id][0].numpy())
        assert np.allclose(compare[str(c_id)].item()["heats"], ref_c[c_id][1].numpy())
