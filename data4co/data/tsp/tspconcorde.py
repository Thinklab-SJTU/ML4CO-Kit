import os
from data4co.utils import download, extract_archive


class TSPConcordeDataset:
    def __init__(self):
        self.url = "https://huggingface.co/datasets/Bench4CO/TSP-Dataset/resolve/main/tspconcorde.tar.gz?download=true"
        self.md5 = "a07aa5586eece38f23ab9ff4e39a1cfd"
        self.dir = "dataset/tspconcorde"
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        if not os.path.exists('dataset/tspconcorde'):
            download(filename="dataset/tspconcorde.tar.gz", url=self.url, md5=self.md5)
            extract_archive(archive_path="dataset/tspconcorde.tar.gz", extract_path=self.dir)
        
    @property
    def supported(self):
        return ["TSP50", "TSP100", "TSP500"]
        