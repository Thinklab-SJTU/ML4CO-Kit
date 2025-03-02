r"""
Download the original dataset of tsplib from hugging face.
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


#############################################
#                  RESOLVE                  #
#############################################

# ATT (only one instance)
TSPLIB_RESOLVE_ATT_PATH = "dataset/tsplib_original/resolved/ATT/problem"
TSPLIB_RESOLVE_ATT_SOLUTION = "dataset/tsplib_original/resolved/ATT/solution"
TSPLIB_RESOLVE_ATT = ["att48"]

# EUC_2D (17 instances)
TSPLIB_RESOLVE_EUC_2D_PATH = "dataset/tsplib_original/resolved/EUC_2D/problem"
TSPLIB_RESOLVE_EUC_2D_SOLUTION = "dataset/tsplib_original/resolved/EUC_2D/solution"
TSPLIB_RESOLVE_EUC_2D = [
    "eil51",
    "berlin52",
    "st70",
    "eil76",
    "pr76",
    "kroA100",
    "kroC100",
    "kroD100",
    "rd100",
    "eil101",
    "lin105",
    "ch130",
    "ch150",
    "tsp225",
    "a280",
    "pr1002",
    "pr2392",
]

# EXPLICIT (8 instances)
TSPLIB_RESOLVE_EXPLICIT_PATH = "dataset/tsplib_original/resolved/EXPLICIT/problem"
TSPLIB_RESOLVE_EXPLICIT_SOLUTION = "dataset/tsplib_original/resolved/EXPLICIT/solution"
TSPLIB_RESOLVE_EXPLICIT = ["gr24", "fri26", "bayg29", "bays29", "gr48", "gr120", "brg180", "pa561"]

# GEO (5 instances)
TSPLIB_RESOLVE_GEO_PATH = "dataset/tsplib_original/resolved/GEO/problem"
TSPLIB_RESOLVE_GEO_SOLUTION = "dataset/tsplib_original/resolved/GEO/solution"
TSPLIB_RESOLVE_GEO = ["ulysses16", "ulysses22", "gr96", "gr202", "gr666"]


###################################################
#                    UNRESOLVE                    #
###################################################

# ATT (only one instance)
TSPLIB_UNRESOLVE_ATT_PATH = "dataset/tsplib_original/unresolved/ATT"
TSPLIB_UNRESOLVE_ATT = ["att532"]

# CEIL_2D (4 instanceds)
TSPLIB_UNRESOLVE_CEIL_2D_PATH = "dataset/tsplib_original/unresolved/CEIL_2D"
TSPLIB_UNRESOLVE_CEIL_2D = ["dsj1000", "pla7397", "pla33810", "pla85900"]

# EUC_2D (61 instances)
TSPLIB_UNRESOLVE_EUC_2D_PATH = "dataset/tsplib_original/unresolved/EUC_2D"
TSPLIB_UNRESOLVE_EUC_2D = [
    "rat99",
    "kroB100",
    "kroE100",
    "pr107",
    "pr124",
    "bier127",
    "pr136",
    "pr144",
    "kroA150",
    "kroB150",
    "pr152",
    "u159",
    "rat195",
    "d198",
    "kroA200",
    "kroB200",
    "ts225",
    "pr226",
    "gil262",
    "pr264",
    "pr299",
    "lin318",
    "linhp318",
    "rd400",
    "fl417",
    "pr439",
    "pcb442",
    "d493",
    "u574",
    "rat575",
    "p654",
    "d657",
    "u724",
    "rat783",
    "u1060",
    "vm1084",
    "pcb1173",
    "d1291",
    "rl1304",
    "rl1323",
    "nrw1379",
    "fl1400",
    "u1432",
    "fl1577",
    "d1655",
    "vm1748",
    "u1817",
    "rl1889",
    "d2103",
    "u2152",
    "u2319",
    "pcb3038",
    "fl3795",
    "fnl4461",
    "rl5915",
    "rl5934",
    "rl11849",
    "usa13509",
    "brd14051",
    "d15112",
    "d18512"
]

# EXPLICIT (6 instances)
TSPLIB_UNRESOLVE_EXPLICIT_PATH = "dataset/tsplib_original/unresolved/EXPLICIT"
TSPLIB_UNRESOLVE_EXPLICIT = ["gr17", "gr21", "dantzig42", "swiss42", "hk48", "brazil58"]

# GEO (2 instances)
TSPLIB_UNRESOLVE_GEO_PATH = "dataset/tsplib_original/unresolved/GEO"
TSPLIB_UNRESOLVE_GEO = ["burma14", "ali535"]


###################################################
#                 TSPLIBOriDataset                #
###################################################

class TSPLIBOriDataset(object):
    r"""
    The class is used to download the original tsplib dataset from hugging face.
    
    ..dropdown:: Example
    
        ::

            >>> from ml4co_kit import TSPLIBOriDataset
            
            #create downloader and load data from huggingface.co
            >>> downloader=TSPLIBOriDataset()
            
            #With the instantiation of the class,the data will be stored in the specified path if the download process is successful.
   
    """
    def __init__(self):
        self.url = "https://huggingface.co/datasets/ML4CO/TSPLIBOriDataset/resolve/main/tsplib_original.tar.gz?download=true"
        self.md5 = "a25f78ef610b6c4ff1cde27f9d5fa6f9"
        self.dir = "dataset/tsplib_original"
        self.raw_data_path = "dataset/tsplib_original.tar.gz"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(self.dir):
            download(filename=self.raw_data_path, url=self.url, md5=self.md5)
            extract_archive(archive_path=self.raw_data_path, extract_path=self.dir)

    @property
    def support(self):
        return {
            "resolved": {
                "ATT": {
                    "path": TSPLIB_RESOLVE_ATT_PATH,
                    "solution": TSPLIB_RESOLVE_ATT_SOLUTION,
                    "problem": TSPLIB_RESOLVE_ATT,
                },
                "EUC_2D": {
                    "path": TSPLIB_RESOLVE_EUC_2D_PATH,
                    "solution": TSPLIB_RESOLVE_EUC_2D_SOLUTION,
                    "problem": TSPLIB_RESOLVE_EUC_2D,
                },
                "EXPLICIT": {
                    "path": TSPLIB_RESOLVE_EXPLICIT_PATH,
                    "solution": TSPLIB_RESOLVE_EXPLICIT_SOLUTION,
                    "problem": TSPLIB_RESOLVE_EXPLICIT,
                },
                "GEO": {
                    "path": TSPLIB_RESOLVE_GEO_PATH,
                    "solution": TSPLIB_RESOLVE_GEO_SOLUTION,
                    "problem": TSPLIB_RESOLVE_GEO,
                },
            },
            "unresolved": {
                "ATT": {
                    "path": TSPLIB_UNRESOLVE_ATT_PATH,
                    "problem": TSPLIB_UNRESOLVE_ATT,
                },
                "EUC_2D": {
                    "path": TSPLIB_UNRESOLVE_EUC_2D_PATH,
                    "problem": TSPLIB_UNRESOLVE_EUC_2D,
                },
                "EXPLICIT": {
                    "path": TSPLIB_UNRESOLVE_EXPLICIT_PATH,
                    "problem": TSPLIB_UNRESOLVE_EXPLICIT,
                },
                "GEO": {
                    "path": TSPLIB_UNRESOLVE_GEO_PATH,
                    "problem": TSPLIB_UNRESOLVE_GEO,
                },
            },
        }
