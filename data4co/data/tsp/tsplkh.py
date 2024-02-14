import os
from data4co.utils import download, extract_archive


class TSPLKHDataset:
    def __init__(self):
        self.url = "https://huggingface.co/datasets/Bench4CO/TSP-Dataset/resolve/main/tsplkh.tar.gz?download=true"
        self.md5 = "8c2bef38b4ede2e500bad96c32543d85"
        self.dir = "dataset/tsplkh"
        self.raw_data_path = "dataset/tsplkh.tar.gz"
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        if not os.path.exists(self.dir):
            download(filename=self.raw_data_path, url=self.url, md5=self.md5)
            extract_archive(archive_path=self.raw_data_path, extract_path=self.dir)
        
    @property
    def supported(self):
        return{
            "TSP50": ["LKH100", "LKH500"],
            "TSP100": ["LKH100", "LKH500", "LKH1000", "LKH5000"],
            "TSP500": ["LKH100", "LKH500", "LKH1000", "LKH5000", 
                      "LKH10000", "LKH50000", "LKH100000"],
        }
        