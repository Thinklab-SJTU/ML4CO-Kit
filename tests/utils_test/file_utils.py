r"""
Test File Utils Module.
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
import shutil
from ml4co_kit import (
    download, pull_file_from_huggingface, get_md5, 
    extract_archive, compress_folder
)


class FileUtilsTester(object):
    """Test cases for file utility functions."""
    
    def __init__(self) -> None:
        pass
    
    def test(self):
        
        ###############################################
        #    Test-1 Download data using directlink    #
        ###############################################

        # 1.1 Directlink and MD5
        file_path = "tmp/tsplib4ml.tar.gz"
        url_link = "https://huggingface.co/datasets/ML4CO/TSPLIB4MLDataset/resolve/main/tsplib4ml.tar.gz"
        md5 = "0696b793c3d53e15b3d95db0a20dcb18"
        
        # 1.2 Using requests to download
        try:
            download(
                file_path=file_path, url=url_link, md5=md5, retries=1)
            os.remove(file_path)
        except:
            pass
        
        # 1.3 Using asyncio to download
        try:
            download(
                file_path=file_path, url=url_link, md5=md5, retries=2)
            os.remove(file_path)
        except:
            pass
        
        # 1.4 Using urllib to download
        try:
            download(
                file_path=file_path, url=url_link, md5=md5, retries=3)
            os.remove(file_path)
        except:
            pass
        
        
        ###############################################
        #    Test-2 Download data from huggingface    #
        ###############################################
        
        # 2.1 Huggingface and MD5
        repo_id = "ML4CO/TSPLIB4MLDataset"
        repo_type = "dataset"
        filename = "tsplib4ml.tar.gz"
        
        # 2.2 Using ``pull_file_from_huggingface`` to download
        pull_file_from_huggingface(
            repo_id=repo_id, 
            repo_type=repo_type, 
            filename=filename, 
            save_path=file_path
        )
        
        # 2.3 Check the MD5 of the downloaded file
        if get_md5(file_path) != md5:
            raise ValueError("Inconsistent MD5 of the downloaded file.")
        
        
        ###############################################
        #    Test-2 Extract the dataset from tar.gz   #
        ###############################################
        
        # 2.1 extract the dataset from tar.gz
        extract_archive(archive_path="tmp/tsplib4ml.tar.gz", extract_path="tmp/tsplib4ml")
        os.remove("tmp/tsplib4ml.tar.gz")
        
        # 2.2 compress the dataset to zip
        compress_folder(folder="tmp/tsplib4ml", compress_path="tmp/tsplib4ml.zip")
        shutil.rmtree("tmp/tsplib4ml")
        
        # 2.3 extract the dataset from zip
        extract_archive(archive_path="tmp/tsplib4ml.zip", extract_path="tmp/tsplib4ml")
        os.remove("tmp/tsplib4ml.zip")
        
        # 2.4 compress the dataset to tar.gz
        compress_folder(folder="tmp/tsplib4ml", compress_path="tmp/tsplib4ml.tar.gz")
        shutil.rmtree("tmp/tsplib4ml")