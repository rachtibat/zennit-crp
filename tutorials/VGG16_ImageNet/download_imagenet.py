import urllib.request
from tqdm import tqdm
from pathlib import Path
from typing import Union

IMAGES_URL = "https://datacloud.hhi.fraunhofer.de/s/RjnK4badZgG7gMq/download"
KIT_URL = "https://datacloud.hhi.fraunhofer.de/s/iz2MYBiGFCpxfg5/download"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(output_path: Union[str, Path]):
    """
    Downloads the ImageNet validation dataset from Fraunhofer HHI server. 
    We provide the dataset only for testing purposes for the tutorial! By downloading you aggree to the terms 
    of access listed on https://image-net.org/download.php.

    Paramters:
    ---------
        output_path: str, pathlib.Path
            parent folder of ImageNet val dataset and development kit.
    """

    if type(output_path) == str:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    print("By downloading you aggree to the terms of access listed on https://image-net.org/download.php")
    print("Start downloading ImageNet development kit")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=KIT_URL.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            KIT_URL, filename=output_path / Path("ILSVRC2012_devkit_t12.tar.gz"), reporthook=t.update_to)

    print("Start downloading ImageNet validation dataset")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=IMAGES_URL.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            IMAGES_URL, filename=output_path / Path("ILSVRC2012_img_val.tar"), reporthook=t.update_to)
