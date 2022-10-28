from pathlib import Path
from PIL import Image
from typing import Any, Tuple, Dict, List


class Cache:
    """
    Abstract class that imlplements the core functionality for caching reference images.
    """

    def __init__(self, path="cache"):

        self.path = Path(path)

    def save(self, ref_c, layer_name, mode, r_range, **kwargs) -> None:

        raise NotImplementedError("'Cache' class must be implemented!")

    def load(self, concept_ids, layer_name, mode, r_range, **kwargs) -> Tuple[Dict[int, Any],
                                                                              Dict[int, Tuple[int, int]]]:

        raise NotImplementedError("'Cache' class must be implemented!")


class ImageCache(Cache):
    """
    Cache that saves single PIL.Image files
    """

    def create_path(self, layer_name, mode, composite, rf, func_name):

        folder_name = mode + "_ " + composite.__class__.__name__
        if rf:
            folder_name += "_rf"

        path = self.path / Path(func_name, folder_name, layer_name)
        path.mkdir(parents=True, exist_ok=True)

        return path

    def save(self, ref_dict: Dict[int, Image.Image],
             layer_name, mode, r_range: Tuple[int, int],
             composite, rf, func_name) -> None:
        """
        Saves PIL.Images inside 'ref_dict' in the path defined by the remaining arguments.

        Parameters:
        ----------
        red_dict: dict of PIL.Image objects
        layer_name: str
        mode: str, 'relevance' or 'activation'
        r_range: tuple (int, int)
        composite: zennit.composites object
        rf: boolean
        func_name: str, 'get_max_reference' or 'get_stats_reference'

        """

        path = self.create_path(layer_name, mode, composite, rf, func_name)

        for id in ref_dict:
            imgs = ref_dict[id]
            for img, r in zip(imgs, range(*r_range)):

                if not isinstance(img, Image.Image):
                    raise TypeError(f"'ImageCache' can only save PIL.Image objects. \
                        But you try to save a {type(img)} object.")

                img.save(path / Path(f"{id}_{r}.png"), optimize=True)

    def load(self, indices: List[int],
             layer_name, mode, r_range, composite, rf, func_name) -> Tuple[Dict[int, Any],
                                                                           Dict[int, Tuple[int, int]]]:
        """
        Loads PIL.Images with concept index 'indices' and layer 'layer_name' from the path defined by the remaining arguments.

        Parameters:
        ----------
        indices: list of int
        layer_name: str
        mode: str, 'relevance' or 'activation'
        r_range: tuple (int, int)
        composite: zennit.composites object
        rf: boolean
        func_name: str, 'get_max_reference' or 'get_stats_reference'

        """

        path = self.create_path(layer_name, mode, composite, rf, func_name)
        ref_c, not_found = {}, {}

        for id in indices:
            ref_c[id] = []

            for r in range(*r_range):

                try:
                    img = Image.open(path / Path(f"{id}_{r}.png"))
                    ref_c[id].append(img)
                except FileNotFoundError:
                    not_found[id] = (r, r_range[-1])
                    break

        return ref_c, not_found
