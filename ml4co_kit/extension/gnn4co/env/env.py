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
import torch
import numpy as np
from typing import Any
from torch.utils.data import DataLoader, Dataset
from ml4co_kit.learning.env import BaseEnv
from ml4co_kit.wrapper.mcl import MClWrapper
from ml4co_kit.wrapper.mis import MISWrapper
from ml4co_kit.wrapper.mvc import MVCWrapper
from ml4co_kit.wrapper.tsp import TSPWrapper
from ml4co_kit.wrapper.mcut import MCutWrapper
from ml4co_kit.wrapper.atsp import ATSPWrapper
from ml4co_kit.wrapper.cvrp import CVRPWrapper
from .denser import GNN4CODenser
from .sparser import GNN4COSparser


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
        task: str = None,
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
        super().__init__(
            name="GNN4COEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device
        )
        
        # basic
        self.task = task
        self.sparse = sparse_factor > 0
        self.sparse_factor = sparse_factor
        
        # train data folder and val path
        self.train_folder = train_folder
        self.val_path = val_path
        
        # ml4co-kit wrapper
        self.atsp_wrapper = ATSPWrapper()
        self.cvrp_wrapper = CVRPWrapper()
        self.mcl_wrapper = MClWrapper()
        self.mcut_wrapper = MCutWrapper()
        self.mis_wrapper = MISWrapper()
        self.mvc_wrapper = MVCWrapper()
        self.tsp_wrapper = TSPWrapper()
        
        # dataset (Fake)
        self.store_data = store_data
        self.train_dataset = FakeDataset(train_data_size)
        self.val_dataset = FakeDataset(val_data_size)
          
        # data_processor (sparser and denser)
        if self.sparse:
            self.data_processor = GNN4COSparser(self.sparse_factor, self.device)
        else:
            self.data_processor = GNN4CODenser(self.device)
        
        # load data
        if self.mode is not None:
            self.load_data()

    def change_device(self, device: str):
        self.device = device
        self.data_processor.device = device
    
    def load_data(self):
        if self.mode == "train":
            self.train_sub_files = [
                os.path.join(self.train_folder, train_files) \
                    for train_files in os.listdir(self.train_folder) 
            ]
            self.train_sub_files_num = len(self.train_sub_files)
            self.train_data_historty_cache = dict()
            self.train_data_cache = None
            self.val_data_cache = None
            self.train_data_cache_idx = 0
        else:
            pass
        
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

    #################################
    #       Generate Val Data       #           
    #################################

    def generate_val_data(self, val_idx: int) -> Any:
        begin_idx = val_idx * self.val_batch_size
        end_idx = begin_idx + self.val_batch_size
        if self.task == "ATSP":
            return self.generate_val_data_atsp(begin_idx, end_idx)
        elif self.task == "CVRP":
            return self.generate_val_data_cvrp(begin_idx, end_idx)
        elif self.task == "MCl":
            return self.generate_val_data_mcl(begin_idx, end_idx)
        elif self.task == "MCut":
            return self.generate_val_data_mcut(begin_idx, end_idx)
        elif self.task == "MIS":
            return self.generate_val_data_mis(begin_idx, end_idx)
        elif self.task == "MVC":
            return self.generate_val_data_mvc(begin_idx, end_idx)
        elif self.task == "TSP":
            return self.generate_val_data_tsp(begin_idx, end_idx) 

    def generate_val_data_atsp(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.atsp_wrapper.from_txt(self.val_path, ref=True)
            self.val_data_cache = self.atsp_wrapper.task_list
        return self.data_processor.atsp_batch_data_process(
            task_data = self.val_data_cache[begin_idx:end_idx]
        )
    
    def generate_val_data_cvrp(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.cvrp_wrapper.from_txt(self.val_path, ref=True)
            self.val_data_cache = self.cvrp_wrapper.task_list
              
        return self.data_processor.cvrp_batch_data_process(
            task_data = self.val_data_cache[begin_idx:end_idx]
        )

    def generate_val_data_mcl(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mcl_wrapper.from_txt(self.val_path, ref=True)
            self.val_data_cache = self.mcl_wrapper.task_list
        return self.data_processor.mcl_batch_data_process(
            task_data = self.val_data_cache[begin_idx:end_idx]
        )

    def generate_val_data_mcut(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mcut_wrapper.from_txt(self.val_path, ref=True)
            self.val_data_cache = self.mcut_wrapper.task_list
        return self.data_processor.mcut_batch_data_process(
            task_data = self.val_data_cache[begin_idx:end_idx]
        )
    
    def generate_val_data_mis(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mis_wrapper.from_txt(self.val_path, ref=True)
            self.val_data_cache = self.mis_wrapper.task_list
        return self.data_processor.mis_batch_data_process(
            task_data = self.val_data_cache[begin_idx:end_idx]
        )
    
    def generate_val_data_mvc(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mvc_wrapper.from_txt(self.val_path, ref=True)
            self.val_data_cache = self.mvc_wrapper.task_list
        return self.data_processor.mvc_batch_data_process(
            task_data = self.val_data_cache[begin_idx:end_idx]
        )

    def generate_val_data_tsp(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.tsp_wrapper.from_txt(self.val_path, ref=True)
            self.val_data_cache = self.tsp_wrapper.task_list
        return self.data_processor.tsp_batch_data_process(
            task_data = self.val_data_cache[begin_idx:end_idx]
        )   
        
    #################################
    #      Generate Train Data      #
    #################################
    
    def generate_train_data(self, batch_size: int) -> Any:
        if self.task == "ATSP":
            return self.generate_train_data_atsp(batch_size)
        elif self.task == "CVRP":
            return self.generate_train_data_cvrp(batch_size)
        elif self.task == "MCl":
            return self.generate_train_data_mcl(batch_size)
        elif self.task == "MCut":
            return self.generate_train_data_mcut(batch_size)
        elif self.task == "MIS":
            return self.generate_train_data_mis(batch_size)
        elif self.task == "MVC":
            return self.generate_train_data_mvc(batch_size)
        elif self.task == "TSP":
            return self.generate_train_data_tsp(batch_size) 

    def generate_train_data_atsp(self, batch_size: int)  -> Any:
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]
            
            # check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else:  
                # load data from the train file
                print(f"\nload atsp train data from {sel_train_sub_file_path}")
                self.atsp_wrapper.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = self.atsp_wrapper.task_list
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        batch_task_data = self.train_data_cache[begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.atsp_batch_data_process(batch_task_data)
    
    def generate_train_data_cvrp(self, batch_size: int) -> Any:
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]
            
            # check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else:
                # load data from the train file
                print(f"\nload cvrp train data from {sel_train_sub_file_path}")
                self.cvrp_wrapper.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = self.cvrp_wrapper.task_list
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        batch_task_data = self.train_data_cache[begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.cvrp_batch_data_process(batch_task_data)

    def generate_train_data_mcl(self, batch_size: int) -> Any:
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]

            # check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else:
                # load data from the train file
                print(f"\nload mcl train data from {sel_train_sub_file_path}")
                self.mcl_wrapper.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = self.mcl_wrapper.task_list
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        batch_task_data = self.train_data_cache[begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.mcl_batch_data_process(batch_task_data)

    def generate_train_data_mcut(self, batch_size: int) -> Any:
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]

            # check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else:
                # load data from the train file
                print(f"\nload mcut train data from {sel_train_sub_file_path}")
                self.mcut_wrapper.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = self.mcut_wrapper.task_list
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        batch_task_data = self.train_data_cache[begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
            
        # sparse process
        return self.data_processor.mcut_batch_data_process(batch_task_data)

    def generate_train_data_mis(self, batch_size: int) -> Any:
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]
            
            # check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else:
                # load data from the train file
                print(f"\nload mis train data from {sel_train_sub_file_path}")
                self.mis_wrapper.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = self.mis_wrapper.task_list
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
            
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        batch_task_data = self.train_data_cache[begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.mis_batch_data_process(batch_task_data)

    def generate_train_data_mvc(self, batch_size: int) -> Any:
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]

            # check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else: 
                # load data from the train file
                print(f"\nload mvc train data from {sel_train_sub_file_path}")
                self.mvc_wrapper.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = self.mvc_wrapper.task_list
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
            
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        batch_task_data = self.train_data_cache[begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.mvc_batch_data_process(batch_task_data)
     
    def generate_train_data_tsp(self, batch_size: int) -> Any:
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]

            # check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else: 
                # load data from the train file
                print(f"\nload tsp train data from {sel_train_sub_file_path}")
                self.tsp_wrapper.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = self.tsp_wrapper.task_list
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
            
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
    
        # retrieve a portion of data from the cache
        batch_task_data = self.train_data_cache[begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
            
        # data process
        return self.data_processor.tsp_batch_data_process(batch_task_data)