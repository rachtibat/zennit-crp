"""
Implements a dynamic dataloader.
"""

from typing import Iterator

from contextlib import contextmanager
from torch.utils.data import Sampler, DataLoader


class DynamicSampler(Sampler[int]):
    """
    Samples elements from a given list of indices.
    """

    def __init__(self) -> None:
        self.indices = []

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
    
    def set_indices(self, indices):
        self.indices = indices

class DynamicLoader(DataLoader):
    """
    A dataloader that can change the indices it samples from.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.sampler, DynamicSampler):
            raise ValueError("Sampler must be an DynamicSampler")
    
    @contextmanager
    def __getitem__(self, idx):
        self.sampler.set_indices(idx)
        yield super().__iter__()
        self.sampler.set_indices([])