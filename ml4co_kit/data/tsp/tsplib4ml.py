r"""
Download tsplib dataset for machine learning from hugging face.
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


###################################################
#              TSPLIB EUC_2D Original             #
###################################################

# EUC_2D (50 instances) [nodes_num <= 1002]
RESOLVED_PROBLEMS = [
    ["eil51", 51],
    ["berlin52", 52],
    ["st70", 70],
    ["eil76", 76],
    ["pr76", 76],
    ["rat99", 99],
    ["kroA100", 100],
    ["kroB100", 100],
    ["kroC100", 100],
    ["kroD100", 100],
    ["kroE100", 100],
    ["rd100", 100],
    ["eil101", 101],
    ["lin105", 105],
    ["pr107", 107],
    ["pr124", 124],
    ["bier127", 127],
    ["ch130", 130],
    ["pr136", 136],
    ["pr144", 144],
    ["ch150", 150],
    ["kroA150", 150],
    ["kroB150", 150],
    ["pr152", 152],
    ["u159", 159],
    ["rat195", 195],
    ["d198", 198],
    ["kroA200", 200],
    ["kroB200", 200],
    ["ts225", 225],
    ["tsp225", 225],
    ["pr226", 226],
    ["gil262", 262],
    ["pr264", 264],
    ["a280", 280],
    ["pr299", 299],
    ["lin318", 318],
    ["rd400", 400],
    ["fl417", 417],
    ["pr439", 439],
    ["pcb442", 442],
    ["d493", 493],
    ["u574", 574],
    ["rat575", 575],
    ["p654", 654],
    ["d657", 657],
    ["u724", 724],
    ["rat783", 783],
    ["pr1002", 1002]
]

# EUC_2D (28 instances) [nodes_num >= 1060]
UNRESOLVED_PROBLEMS = [
    ["u1060", 1060],
    ["vm1084", 1084],
    ["pcb1173", 1173],
    ["d1291", 1291],
    ["rl1304", 1304],
    ["rl1323", 1323],
    ["nrw1379", 1379],
    ["fl1400", 1400],
    ["u1432", 1432],
    ["fl1577", 1577],
    ["d1655", 1655],
    ["vm1748", 1748],
    ["u1817", 1897],
    ["rl1889", 1889],
    ["d2103", 2103],
    ["u2152", 2152],
    ["u2319", 2319],
    ["pr2392", 2392],
    ["pcb3038", 3038],
    ["fl3795", 3795],
    ["fnl4461", 4461],
    ["rl5915", 5915],
    ["rl5934", 5934],
    ["rl11849", 11849],
    ["usa13509", 13509],
    ["brd14051", 14051],
    ["d15112", 15112],
    ["d18512", 18512]
]


###################################################
#                 TSPLIB4MLDataset                #
###################################################

class TSPLIB4MLDataset(object):
    r"""
    The class is used to download the tsplib dataset for machine learning from hugging face.
    
    ..dropdown:: Example
    
        ::

            >>> from ml4co_kit import TSPLIB4MLDataset
            
            #create downloader and load data from huggingface.co
            >>> downloader=TSPLIB4MLDataset()
            
            #With the instantiation of the class,the data will be stored in the specified path if the download process is successful.
   
    """
    def __init__(self):
        self.url = "https://huggingface.co/datasets/ML4CO/TSPLIB4MLDataset/resolve/main/tsplib4ml.tar.gz?download=true"
        self.md5 = "0696b793c3d53e15b3d95db0a20dcb18"
        self.dir = "dataset/tsplib4ml"
        self.raw_data_path = "dataset/tsplib4ml.tar.gz"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(self.dir):
            download(filename=self.raw_data_path, url=self.url, md5=self.md5)
            extract_archive(archive_path=self.raw_data_path, extract_path=self.dir)

    @property
    def support(self):
        return {
            "resolved" : {
                "tsp": "dataset/tsplib4ml/resolved/tsp",
                "txt_raw": "dataset/tsplib4ml/resolved/txt_raw", 
                "txt_normalize": "dataset/tsplib4ml/resolved/txt_normalize",
                "problem": RESOLVED_PROBLEMS
            },
            "unsolved" : {
                "tsp": "dataset/tsplib4ml/unresolved",
                "problem": UNRESOLVED_PROBLEMS,
            }
        }
