import os
from ml4co_kit.utils import download, extract_archive


class SATLIBOriDataset(object):
    def __init__(self) -> None:
        self.url = "https://huggingface.co/datasets/ML4CO/SATLIBOriDataset/resolve/main/satlib_original.tar.gz?download=true"
        self.md5 = "0da8a73e2b79a6b5e6156005959ce509"
        self.dir = "dataset/satlib_original/"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(self.dir):
            download(filename="dataset/satlib_original.tar.gz", url=self.url, md5=self.md5)
            extract_archive(archive_path="dataset/satlib_original.tar.gz", extract_path=self.dir)
