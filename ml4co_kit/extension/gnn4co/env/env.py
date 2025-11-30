r"""
GNN4CO Environment.
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
import copy
import torch
import numpy as np
from typing import Union, Tuple, List
from torch.utils.data import DataLoader, Dataset
from ml4co_kit.learning.env import BaseEnv
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from .denser import GNN4CODenser, DenseDataBatch
from .sparser import GNN4COSparser, SparseDataBatch


class FakeDataset(Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx: int):
        return torch.tensor([idx])


class GNN4COEnv(BaseEnv):
    def __init__(
        self,
        task_type: TASK_TYPE,
        wrapper: WrapperBase,
        mode: str = None,
        train_data_size: int = 128000,
        val_data_size: int = 128,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        sparse_factor: int = 50,
        device: str = "cpu",
        train_folder: str = None,
        val_path: str = None,
        store_data: bool = True,
    ):
        super(GNN4COEnv, self).__init__(
            name="GNN4COEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device
        )
        
        # Basic info
        self.task_type = task_type
        self.val_wrapper = copy.deepcopy(wrapper)
        self.train_wrapper = copy.deepcopy(wrapper)
        self.sparse = sparse_factor > 0
        self.sparse_factor = sparse_factor

        # Val dataset related 
        self.val_path = val_path
        self.val_data_cache = None
        self.val_dataset = FakeDataset(val_data_size)

        # Train dataset related 
        self.store_data = store_data
        self.train_data_cache = None
        self.train_folder = train_folder
        self.train_data_historty_cache = dict()
        self.train_dataset = FakeDataset(train_data_size)
        if self.mode == "train":
            self.train_sub_files = [
                os.path.join(self.train_folder, train_files) \
                    for train_files in os.listdir(self.train_folder) 
            ]
            self.train_sub_files_num = len(self.train_sub_files)
            self.train_data_cache_idx = 0

        # Data processor (sparser and denser)
        if self.sparse:
            self.data_processor = GNN4COSparser(self.sparse_factor, self.device)
        else:
            self.data_processor = GNN4CODenser(self.device)
        
    def train_dataloader(self):
        train_dataloader=DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True, 
            drop_last=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader=DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader=DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return test_dataloader

    def generate_val_data(
        self, val_idx: int
    ) -> Tuple[List[TaskBase], Union[SparseDataBatch, DenseDataBatch]]:
        # Get batch index
        begin_idx = val_idx * self.val_batch_size
        end_idx = begin_idx + self.val_batch_size
        
        # Check cache
        if self.val_data_cache is None:
            if self.val_path.endswith(".txt"):
                self.val_wrapper.from_txt(self.val_path, ref=True, show_time=True)
            elif self.val_path.endswith(".pkl"):
                self.val_wrapper.from_pickle(self.val_path, show_time=True)
            else:
                raise ValueError(f"Unsupported file extension: {self.val_path}")
            self.val_data_cache = self.val_wrapper.task_list
        
        # Get batch_task_data
        batch_task_data = self.val_data_cache[begin_idx:end_idx]
        batch_processed_data = self.data_processor.batch_data_process(batch_task_data)
        return batch_task_data, batch_processed_data
        
    def generate_train_data(
        self, batch_size: int
    ) -> Tuple[List[TaskBase], Union[SparseDataBatch, DenseDataBatch]]:
        # Get batch index
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        
        # Check cache
        if self.train_data_cache is None or end_idx > len(self.train_data_cache):
            # Select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path: str = self.train_sub_files[sel_idx]
            
            # Check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # Using data cache if the data is in the cache
                print(f"\nUsing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else:  
                # Load data from the train file
                print(f"\nLoad train data from {sel_train_sub_file_path}")
                if sel_train_sub_file_path.endswith(".txt"):
                    self.train_wrapper.from_txt(sel_train_sub_file_path, ref=True, show_time=True)
                elif sel_train_sub_file_path.endswith(".pkl"):
                    self.train_wrapper.from_pickle(sel_train_sub_file_path, show_time=True)
                else:
                    raise ValueError(f"Unsupported file extension: {sel_train_sub_file_path}")
                self.train_data_cache = self.train_wrapper.task_list
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # Update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # Get batch task data & Data process
        batch_task_data = self.train_data_cache[begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        batch_processed_data = self.data_processor.batch_data_process(batch_task_data)
        return batch_task_data, batch_processed_data