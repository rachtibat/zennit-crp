import urllib.request
from tqdm import tqdm

IMAGES_URL = "https://datacloud.hhi.fraunhofer.de/s/RjnK4badZgG7gMq/download"
KIT_URL = "https://datacloud.hhi.fraunhofer.de/s/iz2MYBiGFCpxfg5/download"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_imagnet(url, output_path):
    """
    Functionality not tested!
    """
    print("By downloading you aggree to the terms of access listed on https://image-net.org/download.php")
    print("Start downloading ImageNet development kit")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            KIT_URL, filename=output_path, reporthook=t.update_to)

    print("Start downloading ImageNet validation dataset")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            IMAGES_URL, filename=output_path, reporthook=t.update_to)
