import os
import torch
import torch.distributed
from torch import nn
from typing import Optional, List
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
        save_top_k: int = -1,
        mode: str = None,
    ):
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
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
        # logger
        logger: Optional[Logger] = None,
        wandb_logger_name: str = "wandb",
        resume_id: Optional[str] = None,
        # checkpoint
        ckpt_save_path: Optional[str] = None,
        ckpt_monitor: str = "val/loss",
        save_top_k: int = -1,
        mode: str = "min",
        ckpt_every_n_epochs: int = 1,
        ckpt_every_n_train_steps: Optional[int] = None,
        ckpt_filename: str = None,
        # trainer basic
        accelerator: str = "auto",
        strategy: Union[str, Strategy] = None,
        devices: Union[List[int], str, int] = "auto",
        fp16: bool = False,
        max_epochs: int = 100,
        max_steps: int = -1,
        val_check_interval: Optional[int] = None,
        log_every_n_steps: Optional[int] = 50,
        gradient_clip_val: int = 1,
        inference_mode: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        # Disable JIT profiling executor.
        disable_profiling_executor: bool = True,
        # pretrained
        ckpt_path: Optional[str] = None,
        weight_path: Optional[str] = None
    ):
        # logger
        if logger is None:
            self.logger = Logger(name=wandb_logger_name, resume_id=resume_id)
        else:
            self.logger = logger
        
        # checkpoint
        if ckpt_save_path is None:
            self.ckpt_save_path = os.path.join(
                "train_ckpts", self.logger._name, self.logger._id
            )
        self.ckpt_callback = Checkpoint(
            dirpath=self.ckpt_save_path,
            monitor=ckpt_monitor,
            every_n_epochs=ckpt_every_n_epochs,
            every_n_train_steps=ckpt_every_n_train_steps,
            filename="epoch={epoch}-step={step}" if ckpt_filename is None else ckpt_filename,
            save_top_k=save_top_k,
            mode=mode
        )
        
        # learning rate
        self.lr_callback = LearningRateMonitor(logging_interval="step")

        # strategy
        if strategy is None:
            strategy = DDPStrategy(
                static_graph=True,
                find_unused_parameters=True,
                gradient_as_bucket_view=True
            )
            
        # Disable JIT profiling executor
        if disable_profiling_executor:
            try:
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
            except AttributeError:
                pass
        
        # super
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
            gradient_clip_val=gradient_clip_val,
            inference_mode=inference_mode,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs
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
