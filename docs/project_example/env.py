import torch
import torch.utils.data
from .data import TSPDataset
from ml4co_kit.learning.env import BaseEnv
from torch_geometric.data import DataLoader as GraphDataLoader


class TSPEnv(BaseEnv):
    def __init__(
        self,
        nodes_num: int = None,
        mode: str = None,
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        valid_samples: int = 1280,
        num_workers: int = 1,
    ):
        super().__init__(
            name="tsp",
            mode=mode,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
        )
        self.nodes_num = nodes_num
        self.valid_samples = valid_samples
        self.load_data()

    def load_data(self):
        if self.mode == "train":
            self.train_dataset = TSPDataset(
                data_path=self.train_path,
                mode="train",
            )
            self.val_dataset = TSPDataset(
                data_path=self.val_path,
                mode="val",
            )
        elif self.mode == "test":
            self.test_dataset = TSPDataset(
                data_path=self.test_path,
                mode="test",
            )
        else:
            pass

    def train_dataloader(self):
        train_dataloader = GraphDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataset = torch.utils.data.Subset(
            dataset=self.val_dataset, indices=range(self.valid_samples)
        )
        val_dataloader = GraphDataLoader(
            val_dataset, batch_size=self.val_batch_size, shuffle=False
        )
        return val_dataloader

    def test_dataloader(self):
        # force the test batch size to be 1
        self.test_batch_size = 1
        test_dataloader = GraphDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
        )
        return test_dataloader
