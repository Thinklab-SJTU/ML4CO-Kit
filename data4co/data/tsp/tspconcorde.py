import os
from data4co.utils import download, extract_archive


class TSPConcordeDataset:
    def __init__(self):
        self.url = "https://huggingface.co/datasets/Bench4CO/TSP-Dataset/resolve/main/tspconcorde.tar.gz?download=true"
        self.md5 = "a07aa5586eece38f23ab9ff4e39a1cfd"
        self.dir = "dataset/tspconcorde"
        self.raw_data_path = "dataset/tspconcorde.tar.gz"
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        if not os.path.exists(self.dir):
            download(filename=self.raw_data_path, url=self.url, md5=self.md5)
            extract_archive(archive_path=self.raw_data_path, extract_path=self.dir)
        
    @property
    def supported(self):
        supported_files = {
            50: "dataset/tspconcorde/tsp50_concorde_5.68759.txt",
            100: "dataset/tspconcorde/tsp100_concorde_5_7.75585.txt",
            500: "dataset/tspconcorde/tsp500_concorde_16.54581.txt"
        }
        return supported_files
    
    def get_data_path(self, num_nodes: int):
        supported_files = self.supported
        if num_nodes in supported_files.keys():
            return supported_files[num_nodes]
        