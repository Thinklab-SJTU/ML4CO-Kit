import os
from ml4co_kit.utils import download, extract_archive


TSP_UNIFORM_50 = [
    "dataset/tsp_uniform/tsp50_lkh_100_5.68760.txt",
    "dataset/tsp_uniform/tsp50_lkh_500_5.68759.txt",
    "dataset/tsp_uniform/tsp50_concorde_5.68759.txt",
]

TSP_UNIFORM_100 = [
    "dataset/tsp_uniform/tsp100_lkh_100_7.75653.txt",
    "dataset/tsp_uniform/tsp100_lkh_500_7.75598.txt",
    "dataset/tsp_uniform/tsp100_lkh_1k_7.75592.txt",
    "dataset/tsp_uniform/tsp100_lkh_5k_7.75588.txt",
    "dataset/tsp_uniform/tsp100_concorde_5_7.75585.txt",
]

TSP_UNIFORM_500 = [
    "dataset/tsp_uniform/tsp500_lkh_100_16.65813.txt"
    "dataset/tsp_uniform/tsp500_lkh_500_16.59910.txt",
    "dataset/tsp_uniform/tsp500_lkh_5k_16.55766.txt",
    "dataset/tsp_uniform/tsp500_lkh_10k_16.55283.txt",
    "dataset/tsp_uniform/tsp500_lkh_100k_16.54753.txt",
    "dataset/tsp_uniform/tsp500_concorde_16.54581.txt",
]

TSP_UNIFORM_1000 = [
    "dataset/tsp_uniform/tsp1000_concorde_23.11812.txt",
]

TSP_UNIFORM_10000 = [
    "dataset/tsp_uniform/tsp10000_lkh_100k_72.23920.txt",
    "dataset/tsp_uniform/tsp10000_lkh_1m_72.00978.txt",
    "dataset/tsp_uniform/tsp10000_lkh_5m_71.93551.txt",
]


class TSPUniformDataset:
    def __init__(self):
        self.url = "https://huggingface.co/datasets/ML4CO/TSPUniformDataset/resolve/main/tsp_uniform.tar.gz?download=true"
        self.md5 = "494766b3ab67105563e855c7a1f22d80"
        self.dir = "dataset/tsp_uniform"
        self.raw_data_path = "dataset/tsp_uniform.tar.gz"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(self.dir):
            download(filename=self.raw_data_path, url=self.url, md5=self.md5)
            extract_archive(archive_path=self.raw_data_path, extract_path=self.dir)

    @property
    def supported(self):
        supported_files = {
            50: TSP_UNIFORM_50,
            100: TSP_UNIFORM_100,
            500: TSP_UNIFORM_500,
            1000: TSP_UNIFORM_1000,
            10000: TSP_UNIFORM_10000,
        }
        return supported_files
