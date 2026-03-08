r"""
Base class for all datasets in the ML4CO kit.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import shutil
import pathlib
import numpy as np
from typing import Union
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.utils.file_utils import download, extract_archive


class DatasetBase(object):
    """Base class for all datasets."""

    def __init__(
        self, 
        task_type: TASK_TYPE, 
        dataset_name: str,
        precision: Union[np.float32, np.float64] = np.float32,
    ):
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.precision = precision

        # Download related
        self.download_link = f"https://huggingface.co/datasets/ML4CO/ML4CO-Kit/resolve/main/{dataset_name}.zip"
        self.root_dir = pathlib.Path(__file__).parent / "cache"
        self.download_save_path: pathlib.Path = self.root_dir / "raw" / f"{dataset_name}.zip"
        self.extracted_save_path: pathlib.Path = self.root_dir / "extracted" / f"{dataset_name}"
        self.processed_save_path: pathlib.Path = self.root_dir / "processed" / f"{self.task_type.value}_{dataset_name}.pkl"

        # cache
        self.cache: dict = {}
        self.pre_loaded = False

    def _download_extract(self, re_download: bool = False):
        # Clean the download and extracted paths
        if re_download:
            if self.download_save_path.exists():
                os.remove(self.download_save_path)
                shutil.rmtree(self.extracted_save_path)
        
        # Download the dataset
        if not self.download_save_path.exists():
            download(self.download_save_path.as_posix(), self.download_link) 
        
        # Extract the dataset
        if not self.extracted_save_path.exists():
            extract_archive(
                archive_path=self.download_save_path.as_posix(), 
                extract_path=self.extracted_save_path.as_posix()
            )

    def load(
        self, 
        idx: int,
        re_download: bool = False,
        re_process: bool = False,
    ) -> TaskBase:
        # Check if the dataset is preloaded
        if self.pre_loaded == False:
            if self.processed_save_path.exists():
                if re_process:
                    shutil.rmtree(self.processed_save_path)
                else:
                    self.pre_loaded = True

        # Dataset is not preloaded: download, extract, preprocess it
        if self.pre_loaded == False:
            # Download and extract the dataset
            self._download_extract(re_download)

            # Check if the dataset is preprocessed
            if not self.processed_save_path.exists():
                self._preprocess()

            # Set pre_loaded to True
            self.pre_loaded = True

        # Load the dataset
        return self._load(idx)

    def _preprocess(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _load(self, idx: int) -> TaskBase:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __repr__(self) -> str:
        return f"{self.dataset_name}Dataset for {self.task_type.value}"