from data4co.utils import download, extract_archive
from data4co.utils.tsp_utils import get_data_from_tsp_file, generate_opt_tour_file

class TSPLIB:
    def __init__(self):
        self.url = "https://huggingface.co/datasets/Bench4CO/TSP-Dataset/resolve/main/tsplib.tar.gz?download=true"
        self.md5 = "5a57fa47df781a6de898fdbf0b74c86e"
        download(filename="dataset/tsplib.tar.gz", url=self.url, md5=self.md5)
        extract_archive(archive_path="dataset/tsplib.tar.gz", extract_path="dataset/tsplib")
        self.data_process()
        
    def data_process():
        pass
        