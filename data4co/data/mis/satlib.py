import os
import random
import pickle
from tqdm import tqdm
from data4co.utils.mis_utils import sat_to_mis_graph
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
        self.mis_graph = sat_to_mis_graph(self.data_path)


class SATLIBDataset:
    def __init__(self) -> None:
        self.url = "https://huggingface.co/datasets/Bench4CO/SAT-Dataset/resolve/main/satlib.tar.gz?download=true"
        self.md5 = "b83d1d7ca4574d93884c0456a2e90f0c"
        self.dir = "dataset/satlib/"
        self.processed_dir = "dataset/satlib/processed"
        self.raw_dir = "dataset/satlib/raw_data"
        self.test_dir = "dataset/satlib/test_data"
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        if not os.path.exists(self.dir):
            download(filename="dataset/satlib.tar.gz", url=self.url, md5=self.md5)
            extract_archive(archive_path="dataset/satlib.tar.gz", extract_path=self.dir)
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)

    def generate_mis_from_sat(
        self, 
        src: str="dataset/satlib/test_data",
        out: str=None,
        samples_num: int=-1
    ):
        # get the save path
        folder_name = os.path.basename(src)
        if samples_num == -1:
            pickle_name = f"satlib_{folder_name.lower()}.pickle"
            gpickle_dir_name = f"mis_{folder_name.lower()}"
        else:
            pickle_name = f"satlib_{folder_name.lower()}_{samples_num}.pickle"
            gpickle_dir_name = f"mis_{folder_name.lower()}_{samples_num}"
        if out is None:
            processed_pickle_path = os.path.join(self.processed_dir, pickle_name)
            processed_gpickle_path = os.path.join(self.processed_dir, gpickle_dir_name)
        else:
            if not os.path.exists(out):
                os.makedirs(out)
            processed_pickle_path = os.path.join(out, pickle_name)
            processed_gpickle_path = os.path.join(out, gpickle_dir_name)
            
        # check if the processed data exists
        if not os.path.exists(processed_gpickle_path):
            os.mkdir(processed_gpickle_path)
        if os.path.exists(processed_pickle_path):
            with open(processed_pickle_path, 'rb') as f:
                dataset = pickle.load(f)
            return processed_pickle_path, processed_gpickle_path, dataset
        
        # process the cnf data
        dataset = list()
        files = os.listdir(src)
        if samples_num != -1:
            files = random.sample(files, samples_num)
        for file in tqdm(files, desc=f"Processing files in {src}"):
            file_path = os.path.join(src, file)
            mis_graph_path = os.path.join(processed_gpickle_path, file.replace(".cnf", ".gpickle"))
            clause_num = int(file[13:16])
            backbone_size = int(file[18:20])
            sat_data = SATLIBData(
                data_path=file_path,
                variable_num=100,
                clause_num=clause_num,
                backbone_size=backbone_size
            )
            dataset.append(sat_data)
            # write .gpickle
            with open(mis_graph_path, "wb") as f:
                pickle.dump(sat_data.mis_graph, f, pickle.HIGHEST_PROTOCOL)
                
        # write the processed data
        with open(processed_pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
        return processed_pickle_path, processed_gpickle_path, dataset