r"""
The class provided to download dataset of TSP from hugging face.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
from ml4co_kit.utils import download, extract_archive


TSP_UNIFORM_50 = [
    "dataset/tsp_uniform_20240825/tsp50_concorde_5.68759.txt",
    "dataset/tsp_uniform_20240825/tsp50_lkh_500_5.68763.txt",
]

TSP_UNIFORM_100 = [
    "dataset/tsp_uniform_20240825/tsp100_concorde_7.75585.txt",
    "dataset/tsp_uniform_20240825/tsp100_lkh_500_7.75594.txt",
]

TSP_UNIFORM_200 = [
    "dataset/tsp_uniform_20240825/tsp200_concorde_10.71908.txt",
    "dataset/tsp_uniform_20240825/tsp200_lkh_500_10.71931.txt",
]

TSP_UNIFORM_500 = [
    "dataset/tsp_uniform_20240825/tsp500_concorde_16.54581.txt"
    "dataset/tsp_uniform_20240825/tsp500_lkh_500_16.54630.txt",
]

TSP_UNIFORM_1000 = [
    "dataset/tsp_uniform_20240825/tsp1000_concorde_23.11812.txt",
    "dataset/tsp_uniform_20240825/tsp1000_lkh_500_23.11949.txt"
]

TSP_UNIFORM_10000 = [
    "dataset/tsp_uniform_20240825/tsp10000_concorde_large_71.84185.txt",
    "dataset/tsp_uniform_20240825/tsp10000_lkh_500_71.75483.txt",
]


class TSPUniformDataset(object):
    """
    The class is used to download the uniform dataset from hugging face.

    Concorde: Exact
    Concorde: time_limit: 600s
    LKH: 500 trials, 1 runs
    
    ..dropdown:: Example
    
        ::

            >>> from ml4co_kit import TSPUniformDataset
            
            #create downloader and load data from huggingface.co
            >>> downloader=TSPUniformDataset()
            
            #With the instantiation of the class,the data will be stored in the specified path if the download process is successful.
            
    """
    def __init__(self):
        self.url = "https://huggingface.co/datasets/ML4CO/TSPUniformDataset/resolve/main/tsp_uniform_20240825.tar.gz?download=true"
        self.md5 = "44371d7c99b35d77fe18220122c564c1"
        self.dir = "dataset/tsp_uniform_20240825"
        self.raw_data_path = "dataset/tsp_uniform_20240825.tar.gz"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(self.dir):
            download(filename=self.raw_data_path, url=self.url, md5=self.md5)
            extract_archive(archive_path=self.raw_data_path, extract_path=self.dir)

    @property
    def supported(self):
        r"""
        the list of supported filename.
        """
        supported_files = {
            50: TSP_UNIFORM_50,
            100: TSP_UNIFORM_100,
            200: TSP_UNIFORM_200,
            500: TSP_UNIFORM_500,
            1000: TSP_UNIFORM_1000,
            10000: TSP_UNIFORM_10000,
        }
        return supported_files
