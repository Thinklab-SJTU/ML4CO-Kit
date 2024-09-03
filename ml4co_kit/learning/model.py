import torch
import pytorch_lightning as pl
from torch import nn
from typing import Any
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities import rank_zero_info
from ml4co_kit.learning.env import BaseEnv


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        env: BaseEnv,
        model: nn.Module,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
    ):
        super(BaseModel, self).__init__()
        self.env = env
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.num_training_steps_cached = None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def load_data(self):
        self.env.load_data()

    def train_dataloader(self):
        return self.env.train_dataloader()

    def val_dataloader(self):
        return self.env.val_dataloader()

    def test_dataloader(self):
        return self.env.test_dataloader()

    def get_net(self, network_type: str):
        self.network_type = network_type
        return NotImplementedError

    def configure_optimizers(self):
        """ """
        rank_zero_info(
            "Parameters: %d" % sum([p.numel() for p in self.model.parameters()])
        )
        rank_zero_info("Training steps: %d" % self.get_total_num_training_steps())

        if self.lr_scheduler == "constant":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = get_schedule_fn(
                self.lr_scheduler, self.get_total_num_training_steps()
            )(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def get_total_num_training_steps(self) -> int:
        """
        Total training steps inferred from datamodule and devices.
        """
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached

        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches * len(dataset)
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )
        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs

        return self.num_training_steps_cached

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        """Shared step between train/val/test. To be implemented in subclasses."""
        raise NotImplementedError(
            "Shared step is required to implemented in subclasses."
        )

    def training_step(self, batch: Any, batch_idx: int):
        # To use new data every epoch, we need to call reload_dataloaders_every_epoch=True in Trainer
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def solve(self, data: Any, batch_size: int = 16, device="cpu"):
        """solve function, return heatmap"""
        raise NotImplementedError("solve is required to implemented in subclasses.")

    def load_weights(self):
        """load state dict from checkpoint"""
        raise NotImplementedError(
            "``load_ckpt`` is required to implemented in subclasses."
        )


def get_schedule_fn(scheduler, num_training_steps):
    """Returns a callable scheduler_fn(optimizer).
    Todo: Sanitize and unify these schedulers...
    """

    def get_one_cycle(optimizer, num_training_steps):
        """Simple single-cycle scheduler. Not including paper/fastai three-phase things or asymmetry."""

        def lr_lambda(current_step):
            if current_step < num_training_steps / 2:
                return float(current_step / (num_training_steps / 2))
            else:
                return float(2 - current_step / (num_training_steps / 2))

        return LambdaLR(optimizer, lr_lambda, -1)

    if scheduler == "cosine-decay":
        scheduler_fn = partial(
            torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=num_training_steps,
            eta_min=0.0,
        )
    elif scheduler == "one-cycle":  # this is a simplified one-cycle
        scheduler_fn = partial(
            get_one_cycle,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Invalid schedule {scheduler} given.")
    return scheduler_fn
