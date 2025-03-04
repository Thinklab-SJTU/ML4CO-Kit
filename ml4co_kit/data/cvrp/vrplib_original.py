r"""
Download the original dataset of vrplib from hugging face.
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


class VRPLIBOriDataset(object):
    r"""
    The class is used to download the original tsplib dataset from hugging face.
    
    ..dropdown:: Example
    
        ::

            >>> from ml4co_kit import VRPLIBOriDataset
            
            #create downloader and load data from huggingface.co
            >>> downloader=VRPLIBOriDataset()
            
            #With the instantiation of the class,the data will be stored in the specified path if the download process is successful.
   
    """
    def __init__(self) -> None:
        self.url = "https://huggingface.co/datasets/ML4CO/VRPLIBOriDataset/resolve/main/vrplib_original.tar.gz?download=true"
        self.md5 = "7329db3858b318b5ceeab7d0d68f646e"
        self.dir = "dataset/vrplib_original/"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(self.dir):
            download(filename="dataset/vrplib_original.tar.gz", url=self.url, md5=self.md5)
            extract_archive(archive_path="dataset/vrplib_original.tar.gz", extract_path=self.dir)
