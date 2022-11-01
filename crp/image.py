
from typing import Dict, List, Union, Any
from PIL import Image
import torch
from torchvision.transforms.functional import gaussian_blur
import matplotlib.pyplot as plt
import numpy as np

import zennit.image as zimage
from crp.helper import max_norm

@torch.no_grad()
def opaque_img(data_batch, heatmaps, rf=False, alpha=0.3, vis_th=0.2, crop_th=0.1, kernel_size=19, crop_mask=None) -> Image.Image:
    """
    Draws reference images. The function lowers the opacity in regions with relevance lower than max(relevance)*vis_th.
    In addition, the reference image can be cropped where relevance is less than max(relevance)*crop_th by setting 'rf' to True.

    Parameters:
    ----------
    data_batch: torch.Tensor
        original images from dataset without FeatureVisualization.preprocess() applied to it
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    rf: boolean
        Computes the CRP heatmap for a single neuron and hence restricts the heatmap to the receptive field.
        The amount of cropping is further specified by the 'crop_th' argument.
    alpha: between [0 and 1]
        Regulates the transparency in low relevance regions.
    vis_th: between [0 and 1)
        Visualization Threshold: Increases transparency in regions where relevance is smaller than max(relevance)*vis_th.
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th. 
        Cropping is only applied, if receptive field 'rf' is set to True.
    sigma: scalar
        Parameter of the scipy.ndimage.gaussian_filter function used to smooth the CRP heatmap.
        Standard deviation for the Gaussian kernel.
    crop_mask: (optional) numpy.ndarray or tensor with shape=heatmaps.shape
        Binary mask. Its positive values are used for cropping. If None, the 'crop_mask' is constructed by
        the 'heatmaps' using the 'crop_th' argument.

    Returns:
    --------
    image: list of PIL.Image objects
        If 'rf' is True, reference images have different shapes.

    """

    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):

        filtered_heat = max_norm(gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0])
        vis_mask = filtered_heat > vis_th 

        if crop_mask is None:
            crop_mask_ = filtered_heat > crop_th
        elif crop_mask.shape != heatmaps.shape:
            raise ValueError("'crop_mask' must have same shape as 'heatmaps'")

        if rf:
            # all following 'if clauses' are precautions
            rows, columns = torch.where(crop_mask_)
            row1 = rows.min() if len(rows) != 0 else 0
            row2 = rows.max() if len(rows) != 0 else -1
            col1 = columns.min() if len(columns) != 0 else 0
            col2 = columns.max() if len(columns) != 0 else -1

            if (row1 < row2) and (col1 < col2):
                img_t = data_batch[i, ..., row1:row2, col1:col2]
                if img_t.sum() == 0:
                    # do not crop image because cropped image is empty
                    img = data_batch[i]
                else:
                    img = img_t
                    vis_mask = vis_mask[row1:row2, col1:col2]
            else:
                # do not crop image because rf is empty
                img = data_batch[i]
        else:
            img = data_batch[i]

        inv_mask = ~vis_mask
        img = img * vis_mask + img * inv_mask * alpha
        img = zimage.imgify(img.detach().cpu())

        imgs.append(img)

    return imgs


def imgify(image: Union[Image.Image, torch.Tensor, np.ndarray], cmap: str = "bwr", vmin=None, vmax=None, symmetric=True, level=1.0, grid=False, gridfill=None, resize:int=None,
        padding=False) -> Image.Image:
    """
    Convenient wrapper around zennit.image.imgify supporting tensors, numpy arrays and PIL Images. Allows resizing while keeping the aspect
    ratio intact and padding to a square shape.

    Parameters:
    ----------
    image: torch.Tensor, np.ndarray or PIL Image
        With 2 dimensions greyscale, or 3 dimensions with 1 or 3 values in the first or last dimension (color).
    resize: None or int
        If None, no resizing is applied. If int, sets the maximal aspect ratio of the image.
    padding: boolean
        If True, pads the image into a square shape by setting the alpha channel to zero outside the image.
    vmin: float or obj:numpy.ndarray
        Manual minimum value of the array. Overrides the used norm's minimum value.
    vmax: float or obj:numpy.ndarray
        Manual maximum value of the array. Overrides the used norm's maximum value.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will be used to create a palette. The color map will only be applied for arrays with only a single color channel. The color will be specified as a palette in the PIL Image.
    
    Returns:
    --------
    image: PIL.Image object
    """

    if isinstance(image, torch.Tensor):
        img = zimage.imgify(image.detach().cpu(), cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, level=level, grid=grid, gridfill=gridfill)
    elif isinstance(image, np.ndarray):
        img = zimage.imgify(image, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, level=level, grid=grid, gridfill=gridfill)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError("Only PIL.Image, torch.Tensor or np.ndarray types are supported!")

    if resize:
        ratio = resize/max(img.size)
        new_size = tuple([int(x*ratio) for x in img.size])
        img = img.resize(new_size, Image.NEAREST)

    if padding:
        max_size = max(img.size)
        new_im = Image.new("RGBA", (max_size, max_size))
        new_im.putalpha(0)
        new_im.paste(img, ((max_size-img.size[0])//2, (max_size-img.size[1])//2))
        img = new_im

    return img


def plot_grid(ref_c: Dict[int, List], cmap="bwr", vmin=None, vmax=None, symmetric=False, resize=None, padding=True, figsize=None) -> None:
    """
    Plots lists of images or arrays inside a dictionary. To every element in the list crp.imgify is applied with its respective argument values.

    Parameters:
    ----------
    ref_c: dict with integer keys and list values filled with torch.Tensor, np.ndarray or PIL Image
        To every element in the list crp.imgify is applied.
    resize: None or int
        If None, no resizing is applied. If int, sets the maximal aspect ratio of the image.
    padding: boolean
        If True, pads the image into a square shape by setting the alpha channel to zero outside the image.
    vmin: float or obj:numpy.ndarray
        Manual minimum value of the array. Overrides the used norm's minimum value.
    vmax: float or obj:numpy.ndarray
        Manual maximum value of the array. Overrides the used norm's maximum value.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will be used to create a palette. The color map will only be applied for arrays with only a single color channel. The color will be specified as a palette in the PIL Image.
    
    Returns:
    --------
    matplotlib.pyplot plot
    """

    nrows = len(ref_c)
    ncols = len(next(iter(ref_c.values())))

    plt.close()
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    plt.subplots_adjust(wspace=0.05)

    for r, key in enumerate(ref_c):

        for c, img in enumerate(ref_c[key]):

            img = imgify(img, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, resize=resize, padding=padding)

            if len(axs.shape) == 1:
                ax = axs[c]
            else:
                ax = axs[r, c]

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(key)
