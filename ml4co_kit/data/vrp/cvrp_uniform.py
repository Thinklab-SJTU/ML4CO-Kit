import os
from ml4co_kit.utils import download, extract_archive


CVRP_UNIFORM_20 = [
    "dataset/cvrp_uniform_20240826/cvrp20_hgs_1s_6.13013.txt",
    "dataset/cvrp_uniform_20240826/cvrp20_lkh_500_1runs_6.13560.txt",
    "dataset/cvrp_uniform_20240826/cvrp20_lkh_500_1runs_6.13560.txt",
]

CVRP_UNIFORM_50 = [
    "dataset/cvrp_uniform_20240826/cvrp50_hgs_1s_10.36646.txt",
    "dataset/cvrp_uniform_20240826/cvrp50_lkh_500_1runs_10.40205.txt",
    "dataset/cvrp_uniform_20240826/cvrp50_lkh_500_10runs_10.37266.txt",
]

CVRP_UNIFORM_100 = [
    "dataset/cvrp_uniform_20240826/cvrp100_hgs_20s_15.56319.txt",
    "dataset/cvrp_uniform_20240826/cvrp100_lkh_500_1runs_15.74601.txt",
    "dataset/cvrp_uniform_20240826/cvrp100_lkh_500_10runs_15.61939.txt",
]

CVRP_UNIFORM_1000 = [
    "dataset/cvrp_uniform_20240826/cvrp1000_hgs_43.45630.txt",
]

CVRP_UNIFORM_2000 = [
    "dataset/cvrp_uniform_20240826/cvrp2000_hgs_59.65577.txt",
]

CVRP_UNIFORM_5000 = [
    "dataset/cvrp_uniform_20240826/cvrp5000_hgs_128.52305.txt",
]


class CVRPUniformDataset(object):
    """
    CVRP20, CVRP50, CVRP100:
        10K instances
        LKH Params:
            500 trials (1 runs and 10 runs); 
        HGS Time Limit: 
            20: 1s,
            50: 1s,
            100: 20s
    CVRP1K, CVRP2K, CVRP5K:
        HGS: Not using genetic algorithm, only conducting preliminary local search
            
    """
    def __init__(self):
        self.url = "https://huggingface.co/datasets/ML4CO/CVRPUniformDataset/resolve/main/cvrp_uniform_20240826.tar.gz?download=true"
        self.md5 = "b8c4252725dd1e506178d93bb49ea755"
        self.dir = "dataset/cvrp_uniform_20240826"
        self.raw_data_path = "dataset/cvrp_uniform_20240826.tar.gz"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(self.dir):
            download(filename=self.raw_data_path, url=self.url, md5=self.md5)
            extract_archive(archive_path=self.raw_data_path, extract_path=self.dir)

    @property
    def supported(self):
        supported_files = {
            20: CVRP_UNIFORM_20,
            50: CVRP_UNIFORM_50,
            100: CVRP_UNIFORM_100,
            1000: CVRP_UNIFORM_1000,
            2000: CVRP_UNIFORM_2000,
            5000: CVRP_UNIFORM_5000,
        }
        return supported_files
