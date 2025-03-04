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
from ml4co_kit.utils.file_utils import _get_md5


class ML4TSPDataset(object):
    r"""
    This class provides a basic framwork to download the data of TSP from hugging face.
    """
    def __init__(self):
        self.supported = {
            "tsp50_uniform_lkh_5k_1.28m.txt": "d049c9e86d7b26d2e6d25d896e0545fd",
            "tsp100_uniform_lkh_5k_1.28m.txt": "787bb9c5481cfa7835d640b702f9fe82",
            "tsp500_uinform_concorde_80k.txt": "da01970bdb52aa7874f92222f09b0296", 
            "tsp500_uniform_lkh_50k_128k.txt": "833ef74de9d982b8f7420b001dd628fa",
            "tsp1000_uniform_lkh_100k_64k.txt": "96ff21d0fcf75a7cc9dd2e4fbf8f8789",
            "tsp10000_uniform_concorde_large_6.4k.txt": "3e0e379ca6b49f30e2bad935176cea02"
        }
        self.dir = "dataset/ml4tsp"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def download(
        self,
        filename: str,
        hf_token: str
    ):
        r"""
        Download TSP data from hugging face using the hugging face token and check the downloaded file.

        :param filename: string, the filename of data want to be downloaded.
        :param filename: string, the token used to download the data.

        ..note::
            -the supported file name is "tsp50_uniform_lkh_5k_1.28m.txt",
            "tsp100_uniform_lkh_5k_1.28m.txt",
            "tsp500_uinform_concorde_80k.txt", 
            "tsp500_uniform_lkh_50k_128k.txt",
            "tsp1000_uniform_lkh_100k_64k.txt",
            "tsp10000_uniform_concorde_large_6.4k.txt".

        ..dropdown:: Example

            ::

                >>> from ml4co_kit import ML4TSPDataset

                #create the data_loader 
                >>> data_loader = ML4TSPDataset()

                #download data from hugging face 
                >>> download()
        """
        # check necessary package
        try:
            from huggingface_hub import HfApi
        except:
            raise ModuleNotFoundError("huggingface_hub need be installed!")
        
        # use huggingface token to download
        hf_api = HfApi(token=hf_token)
        hf_api.hf_hub_download(
            repo_id="ML4CO/ML4TSPDataset",
            repo_type="dataset",
            filename=filename,
            cache_dir="hf_cache",
            local_dir=self.dir,
            local_dir_use_symlinks=False
        )
        
        # check the downloaded file (md5)
        self.save_path = os.path.join(self.dir, filename)
        md5 = _get_md5(self.save_path)
        if md5 != self.supported[filename]:
            os.remove(self.save_path)
            raise ValueError("Warning: MD5 check failed for the downloaded content")
            