import os
import torch
from torch import nn
from typing import Optional
from wandb.util import generate_id
from typing import Union, Optional
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import Strategy, DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import WandbLogger


class Checkpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: str = "wandb/checkpoints",
        monitor: str = "val/loss",
        every_n_epochs: int = 1,
        every_n_train_steps=None,
        filename=None,
    ):
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode="min",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps,
            filename=filename,
            auto_insert_metric_name=False,
        )


class Logger(WandbLogger):
    def __init__(
        self,
        name: str = "wandb",
        project: str = "project",
        entity: Optional[str] = None,
        save_dir: str = "log",
        id: Optional[str] = None,
        resume_id: Optional[str] = None,
    ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if id is None and resume_id is None:
            wandb_id = os.getenv("WANDB_RUN_ID") or generate_id()
        else:
            wandb_id = id if id is not None else resume_id

        super().__init__(
            name=name, project=project, entity=entity, save_dir=save_dir, id=wandb_id
        )


class Trainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        logger: Optional[Logger] = None,
        wandb_logger_name: str = "wandb",
        resume_id: Optional[str] = None,
        ckpt_save_path: Optional[str] = None,
        monitor: str = "val/loss",
        every_n_epochs: int = 1,
        every_n_train_steps: Optional[int] = None,
        val_check_interval: Optional[int] = None,
        log_every_n_steps: Optional[int] = 50,
        inference_mode: bool = False,
        accelerator: str = "auto",
        strategy: Union[str, Strategy] = DDPStrategy(static_graph=True),
        max_epochs: int = 100,
        max_steps: int = -1,
        fp16: bool = False,
        ckpt_path: Optional[str] = None,
        weight_path: Optional[str] = None,
        **kwargs,
    ):
        if logger is None:
            self.logger = Logger(name=wandb_logger_name, resume_id=resume_id)
        else:
            self.logger = logger
        if ckpt_save_path is None:
            self.ckpt_save_path = os.path.join(
                "train_ckpts", self.logger._name, self.logger._id
            )
        self.ckpt_callback = Checkpoint(
            dirpath=self.ckpt_save_path,
            monitor=monitor,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps,
            filename="epoch={epoch}-step={step}",
        )
        self.lr_callback = LearningRateMonitor(logging_interval="step")

        devices = torch.cuda.device_count() if torch.cuda.is_available() else "auto"
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=16 if fp16 else 32,
            logger=self.logger,
            callbacks=[
                TQDMProgressBar(refresh_rate=20),
                self.ckpt_callback,
                self.lr_callback,
            ],
            max_epochs=max_epochs,
            max_steps=max_steps,
            check_val_every_n_epoch=1,
            val_check_interval=val_check_interval,
            log_every_n_steps=log_every_n_steps,
            inference_mode=inference_mode,
        )
        if ckpt_path is not None:
            model.load_from_checkpoint(ckpt_path)
        elif weight_path is not None:
            model.load_state_dict(torch.load(weight_path))

        self.train_model = model

    def model_train(self):
        rank_zero_info(
            f"Logging to {self.logger.save_dir}/{self.logger.name}/{self.logger.version}"
        )
        rank_zero_info(f"checkpoint_callback's dirpath is {self.ckpt_save_path}")
        rank_zero_info(f"{'-' * 100}\n" f"{str(self.train_model)}\n" f"{'-' * 100}\n")
        self.fit(self.train_model)
        self.logger.finalize("success")

    def model_test(self):
        rank_zero_info(
            f"Logging to {self.logger.save_dir}/{self.logger.name}/{self.logger.version}"
        )
        rank_zero_info(f"{'-' * 100}\n" f"{str(self.train_model)}\n" f"{'-' * 100}\n")
        self.test(self.train_model)
