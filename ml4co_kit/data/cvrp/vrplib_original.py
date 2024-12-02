import os
from ml4co_kit.utils import download, extract_archive


class VRPLIBOriDataset(object):
    def __init__(self) -> None:
        self.url = "https://huggingface.co/datasets/ML4CO/VRPLIBOriDataset/resolve/main/vrplib_original.tar.gz?download=true"
        self.md5 = "7329db3858b318b5ceeab7d0d68f646e"
        self.dir = "dataset/vrplib_original/"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(self.dir):
            download(filename="dataset/vrplib_original.tar.gz", url=self.url, md5=self.md5)
            extract_archive(archive_path="dataset/vrplib_original.tar.gz", extract_path=self.dir)
