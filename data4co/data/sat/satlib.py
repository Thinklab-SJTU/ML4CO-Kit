import os
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from data4co.utils.sat_utils import CNF
from data4co.utils import download, extract_archive


class SATLIBData:
    def __init__(
        self,
        data_path: str,
        variable_num: int,
        clause_num: int,
        backbone_size: int
    ):
        self.data_path = data_path
        self.variable_num = variable_num
        self.clause_num = clause_num
        self.backone_size = backbone_size
        self.graph = self.to_mis_graph()
        
    def to_mis_graph(self):
        cnf = CNF(self.data_path)
        nv = cnf.nv
        clauses = list(filter(lambda x: x, cnf.clauses))
        ind = { k:[] for k in np.concatenate([np.arange(1, nv+1), -np.arange(1, nv+1)]) }
        edges = []
        for i, clause in enumerate(clauses):
            a = clause[0]
            b = clause[1]
            c = clause[2]
            aa = 3 * i + 0
            bb = 3 * i + 1
            cc = 3 * i + 2
            ind[a].append(aa)
            ind[b].append(bb)
            ind[c].append(cc)
            edges.append((aa, bb))
            edges.append((aa, cc))
            edges.append((bb, cc))
        for i in np.arange(1, nv+1):
            for u in ind[i]:
                for v in ind[-i]:
                    edges.append((u, v))
        graph = nx.from_edgelist(edges)
        return graph


class SATLIBDataset:
    def __init__(self) -> None:
        self.url = "https://huggingface.co/datasets/Bench4CO/SAT-Dataset/resolve/main/satlib.tar.gz?download=true"
        self.md5 = "0da8a73e2b79a6b5e6156005959ce509"
        self.dir = "dataset/satlib/"
        self.processed_dir = "dataset/satlib/processed"
        self.test_dir = "dataset/satlib/test_files"
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        if not os.path.exists(self.dir):
            download(filename="dataset/satlib.tar.gz", url=self.url, md5=self.md5)
            extract_archive(archive_path="dataset/satlib.tar.gz", extract_path=self.dir)
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)

    def get_data_from_folder(self, folder: str="test_files"):
        if folder.startswith(self.dir):
            folder = folder.replace(self.dir, "")
        folder_path = os.path.join(self.dir, folder)
        processed_data_path = os.path.join(self.processed_dir, \
            f"satlib_{folder.lower()}.pickle")
        if os.path.exists(processed_data_path):
            with open(processed_data_path, 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        dataset = list()
        files = os.listdir(folder_path)
        for file in tqdm(files, desc=f"Processing files in {folder_path}"):
            file_path = os.path.join(folder_path, file)
            clause_num = int(file[13:16])
            backbone_size = int(file[18:20])
            sat_data = SATLIBData(
                data_path=file_path,
                variable_num=100,
                clause_num=clause_num,
                backbone_size=backbone_size
            )
            dataset.append(sat_data)
        # write the processed data
        with open(processed_data_path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset