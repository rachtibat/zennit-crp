from pathlib import Path
from PIL import Image
from typing import Any, Tuple, Dict, List, Union


class Cache:
    """
    Abstract class that imlplements the core functionality for caching reference images.
    """

    def __init__(self, path="cache"):

        self.path = Path(path)

    def save(self, ref_c, layer_name, mode, r_range, composite, rf, f_name, plot_name, **kwargs) -> None:

        raise NotImplementedError("'Cache' class must be implemented!")
     
    def load(self, concept_ids, layer_name, mode, r_range, composite, rf, f_name, plot_name, **kwargs) -> Tuple[Dict[int, Any],
                                                                              Dict[int, Tuple[int, int]]]:

        raise NotImplementedError("'Cache' class must be implemented!")

    def extend_dict(self, ref_original, rf_addition):

        raise NotImplementedError("'Cache' class must be implemented!")


class ImageCache(Cache):
    """
    Cache that saves lists or tuple of lists of PIL.Image files that are values of a dictionary.

    Parameters:
    ----------
    path: str or pathlib.Path
        folder where to solve the images

    """

    def _create_path(self, layer_name, mode, composite, rf, func_name, plot_name):

        folder_name = mode + "_" + composite.__class__.__name__ 
        if rf:
            folder_name += "_rf"

        path = self.path / Path(func_name, plot_name, folder_name, layer_name)
        path.mkdir(parents=True, exist_ok=True)

        return path

    def _save_img_list(self, img_list, id, tuple_index, r_range, path):

        for img, r in zip(img_list, range(*r_range)):

            if not isinstance(img, Image.Image):
                raise TypeError(f"'ImageCache' can only save PIL.Image objects. \
                    But you tried to save a {type(img)} object.")

            img.save(path / Path(f"{id}_{tuple_index}_{r}.png"), optimize=True)


    def save(self, ref_dict: Dict[Any, Union[Image.Image, List]],
             layer_name, mode, r_range: Tuple[int, int],
             composite, rf, func_name, plot_name) -> None:
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
        plot_name: str
            name of plot_fn

        """

        path = self._create_path(layer_name, mode, composite, rf, func_name, plot_name)

        for id in ref_dict:
            value = ref_dict[id]

            if isinstance(value, Tuple):

                self._save_img_list(value[0], id, 0, r_range, path)
                self._save_img_list(value[1], id, 1, r_range, path)

            elif isinstance(value[0], Image.Image):

                self._save_img_list(value, id, 0, r_range, path)

    def _load_img_list(self, id, tuple_index, r_range, path):
        
        imgs, not_found = [], None
        for r in range(*r_range):
        
            try:
                img = Image.open(path / Path(f"{id}_{tuple_index}_{r}.png"))
                imgs.append(img)
            except FileNotFoundError:
                not_found = (r, r_range[-1])
                break
        
        return imgs, not_found

    def load(self, indices: List,
             layer_name, mode, r_range, composite, rf, func_name, plot_name) -> Tuple[Dict[Any, Any],
                                                                           Dict[int, Tuple[int, int]]]:
        """
        Loads PIL.Images with concept index 'indices' and layer 'layer_name' from the path defined by the remaining arguments.

        Parameters:
        ----------
        indices: list of int or str
        layer_name: str
        mode: str, 'relevance' or 'activation'
        r_range: tuple (int, int)
        composite: zennit.composites object
        rf: boolean
        func_name: str, 'get_max_reference' or 'get_stats_reference'

        """

        path = self._create_path(layer_name, mode, composite, rf, func_name, plot_name)
        ref_c, not_found = {}, {}

        for id in indices:

            imgs_0, not_found_0 = self._load_img_list(id, 0, r_range, path)
            imgs_1, _ = self._load_img_list(id, 1, r_range, path)

            if imgs_0:

                if imgs_1:
                    # tuple per sample exists
                    ref_c[id] = (imgs_0, imgs_1)
                else:
                    ref_c[id] = imgs_0

            if not_found_0:
                not_found[id] = not_found_0


        return ref_c, not_found


    def extend_dict(self, ref_original, rf_addition):

        for key, value in rf_addition.items():

            if key in ref_original:

                if isinstance(value, Tuple):
                    ref_original[key][0].extend(value[0]) 
                    ref_original[key][1].extend(value[1]) 
                elif isinstance(value, List):
                    ref_original[key].extend(value)
                else:
                    raise TypeError("'ref_original' must contain tuples or a list")
            else:
                ref_original[key] = value

        return ref_original
