
from PIL import Image, ImageOps
import torch
from typing import Dict, List
import zennit.image as zimage

import matplotlib.pyplot as plt


def imgify(
        tensor: torch.Tensor, cmap: str = "bwr", vmin=None, vmax=None, symmetric=True, resize=(224, 224),
        padding=True) -> Image:

    img = zimage.imgify(tensor.detach().cpu(), cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric)

    if resize:
        ratio = max(resize)/max(img.size)
        new_size = tuple([int(x*ratio) for x in img.size])

        img = img.resize(new_size, Image.NEAREST)

    if resize:
        ratio = max(resize)/max(img.size)
        new_size = tuple([int(x*ratio) for x in img.size])

        img = img.resize(new_size, Image.NEAREST)

        if padding:
            new_im = Image.new("RGBA", (resize[0], resize[1]))
            new_im.putalpha(0)
            new_im.paste(img, ((resize[0]-new_size[0])//2, (resize[1]-new_size[1])//2))

            return new_im

    return img


def plot_grid(ref_c: Dict[int, List[torch.Tensor]], cmap="bwr", vmin=None, vmax=None, symmetric=False, resize=(224, 224), padding=True, figsize=None) -> None:

    nrows = len(ref_c)
    ncols = len(next(iter(ref_c.values())))

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
