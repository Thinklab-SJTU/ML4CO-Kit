import os
import pickle
import numpy as np
from tqdm import tqdm
from data4co.utils import download, extract_archive
from data4co.utils.tsp_utils import get_data_from_tsp_file, get_tour_from_tour_file

        
class TSPLIBData:
    def __init__(
        self, 
        name: str, 
        edge_weight_type: str,
        node_coords: np.ndarray=None,
        edge_weight: np.ndarray=None,
        tour: np.ndarray=None
    ):
        self.name = name
        self.edge_weight_type = edge_weight_type
        self.node_coords = node_coords
        self.edge_weight = edge_weight
        self.tour = tour
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class TSPLIBDataset:
    def __init__(self):
        self.url = "https://huggingface.co/datasets/Bench4CO/TSP-Dataset/resolve/main/tsplib.tar.gz?download=true"
        self.md5 = "9ee214be3ad818b60c137d3f9869151b"
        self.dir = "dataset/tsplib"
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        if not os.path.exists('dataset/tsplib'):
            download(filename="dataset/tsplib.tar.gz", url=self.url, md5=self.md5)
            extract_archive(archive_path="dataset/tsplib.tar.gz", extract_path=self.dir)
        if os.path.exists('dataset/tsplib/data.pickle'):
            with open('dataset/tsplib/data.pickle', 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data_process()
            with open('dataset/tsplib/data.pickle', 'wb') as f:
                pickle.dump(self.data, f)

    def data_process(self):
        self.data = dict()
        edge_weight_types = ["EUC_2D", "EXPLICIT", "GEO"]

        # resolved
        self.resolved_euc2d_path = os.path.join(self.dir, "resolved/EUC_2D")
        self.resolved_explicit_path = os.path.join(self.dir, "resolved/EXPLICIT")
        self.resolved_geo_path = os.path.join(self.dir, "resolved/GEO")
        resolved_paths = [self.resolved_euc2d_path,
                        self.resolved_explicit_path,
                        self.resolved_geo_path]
        for edge_weight_type, resolved_path in zip(edge_weight_types, resolved_paths):
            tsp_files = os.listdir(os.path.join(resolved_path, "problem"))
            for tsp_file in tqdm(tsp_files, desc=f"Processing {resolved_path}"):
                tsp_path = os.path.join(resolved_path, f"problem/{tsp_file}")
                tour_path = os.path.join(resolved_path, f"solution/{tsp_file.replace('.tsp', '.opt.tour')}")
                name = tsp_file[:-4]
                node_coords, edge_weight = get_data_from_tsp_file(tsp_path)
                tour = get_tour_from_tour_file(tour_path)
                tsp_data = TSPLIBData(
                    name=name,
                    edge_weight_type=edge_weight_type,
                    node_coords=node_coords,
                    edge_weight=edge_weight,
                    tour=tour
                )
                self.data[name] = tsp_data

        # unresolved
        self.unresolved_euc2d_path = os.path.join(self.dir, "unresolved/EUC_2D")
        self.unresolved_explicit_path = os.path.join(self.dir, "unresolved/EXPLICIT")
        self.unresolved_geo_path = os.path.join(self.dir, "unresolved/GEO")
        unresolved_paths = [self.unresolved_euc2d_path,
                            self.unresolved_explicit_path,
                            self.unresolved_geo_path]
        for edge_weight_type, unresolved_path in zip(edge_weight_types, unresolved_paths):
            tsp_files = os.listdir(unresolved_path)
            for tsp_file in tqdm(tsp_files, desc=f"Processing {unresolved_path}"):
                tsp_path = os.path.join(unresolved_path, tsp_file)
                name = tsp_file[:-4]
                node_coords, edge_weight = get_data_from_tsp_file(tsp_path)
                tsp_data = TSPLIBData(
                    name=name,
                    edge_weight_type=edge_weight_type,
                    node_coords=node_coords,
                    edge_weight=edge_weight,
                )
                self.data[name] = tsp_data
                